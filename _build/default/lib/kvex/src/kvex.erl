%%% @doc
%%% kvex — pure Erlang approximate k-NN vector search on the BEAM.
%%%
%%% Two ETS tables per index:
%%%   • vec table  — `{Id, F32Bin, BinVec}' per vector (source of truth)
%%%   • flat cache — `{flat, F32FlatBin, BvecFlatBin, IdsTuple}' rebuilt on
%%%                  every insert; a single refc-binary per flat, so search
%%%                  never iterates over individual Erlang terms.
%%%
%%% Search path (two sied NIF calls on flat binaries, no Erlang list work):
%%%   1. `sied:hamming_topk_flat/4'   — SIMD POPCNT on BvecFlat, O(N)+O(K logK)
%%%   2. `sied:dot_product_topk_flat/4' — SIMD dot-product on F32Flat candidates
%%%
%%% == Quick start ==
%%% ```
%%% {ok, Ix} = kvex:new(128),
%%% Vec      = [rand:uniform() || _ <- lists:seq(1, 128)],
%%% ok       = kvex:add(Ix, 42, Vec),
%%% {ok, Rs} = kvex:search(Ix, Vec, 5),
%%% ok       = kvex:delete(Ix).
%%% '''
%%% @end
-module(kvex).

-export([version/0, new/1, new/2, delete/1, size/1,
         add/3, add_batch/2, search/3,
         normalize/1, cosine_search/3,
         dump/2, load/1]).

-define(OVERSAMPLE, 10).

%% flat cache key stored in the vec table (atom, never clashes with id())
-define(FLAT_KEY, '$kvex_flat').

-opaque index() :: #{table := ets:tid(), dim := pos_integer()}.
-type opts()    :: #{bits => 2 | 3 | 4}.
-type id()      :: non_neg_integer() | binary().
-type vector()  :: [float()] | binary().

-export_type([index/0, opts/0, id/0, vector/0]).

%%%===================================================================
%%% Public API
%%%===================================================================

-spec version() -> binary().
version() -> <<"0.2.1">>.

-spec new(Dim :: pos_integer()) -> {ok, index()} | {error, term()}.
new(Dim) -> new(Dim, #{}).

-spec new(Dim :: pos_integer(), opts()) -> {ok, index()} | {error, term()}.
%% @doc Creates an empty index for vectors of dimension `Dim'.
new(Dim, _Opts) when is_integer(Dim), Dim > 0 ->
    Tid = ets:new(kvex, [set, protected]),
    ets:insert(Tid, {?FLAT_KEY, <<>>, <<>>, {}}),
    {ok, #{table => Tid, dim => Dim}};
new(Dim, _Opts) ->
    {error, {bad_dim, Dim}}.

-spec delete(index()) -> ok.
delete(#{table := Tid}) ->
    ets:delete(Tid),
    ok.

-spec size(index()) -> non_neg_integer().
size(#{table := Tid}) ->
    ets:info(Tid, size) - 1.   % subtract the flat_cache sentinel

-spec add(index(), id(), vector()) -> ok | {error, term()}.
%% @doc Inserts a single vector. Rebuilds the flat cache — O(N) copy.
add(#{table := Tid, dim := Dim}, Id, Vec0) ->
    F32Bin = to_f32_bin(Vec0),
    case byte_size(F32Bin) div 4 of
        Dim ->
            {ok, BinVec} = sied:to_binary_f32_bin(F32Bin),
            ets:insert(Tid, {Id, F32Bin, BinVec}),
            rebuild_flat(Tid),
            ok;
        Got ->
            {error, {dim_mismatch, Dim, Got}}
    end.

-spec add_batch(index(), [{id(), vector()}]) -> ok | {error, term()}.
%% @doc Inserts vectors in batch. Builds the flat binary incrementally — O(batch).
add_batch(#{table := Tid, dim := Dim}, Pairs) when is_list(Pairs) ->
    case build_entries(Pairs, Dim, 0, [], [], [], []) of
        {ok, Entries, F32New, BvecNew, IdsNew} ->
            ets:insert(Tid, Entries),
            [{?FLAT_KEY, F32Old, BvecOld, IdsOldT}] = ets:lookup(Tid, ?FLAT_KEY),
            IdsOld   = tuple_to_list(IdsOldT),
            F32Flat  = <<F32Old/binary,  F32New/binary>>,
            BvecFlat = <<BvecOld/binary, BvecNew/binary>>,
            IdsTuple = list_to_tuple(IdsOld ++ IdsNew),
            ets:insert(Tid, {?FLAT_KEY, F32Flat, BvecFlat, IdsTuple}),
            ok;
        {error, _} = Err ->
            Err
    end.

-spec search(index(), Query :: vector(), K :: pos_integer()) ->
        {ok, [{id(), Score :: float()}]} | {error, term()}.
search(#{table := Tid, dim := Dim}, Query0, K)
        when is_integer(K), K > 0 ->
    QBin = to_f32_bin(Query0),
    case byte_size(QBin) div 4 of
        Dim ->
            [{?FLAT_KEY, F32Flat, BvecFlat, IdsTuple}] = ets:lookup(Tid, ?FLAT_KEY),
            N = tuple_size(IdsTuple),
            case N of
                0 -> {error, empty_index};
                _ -> do_search(QBin, F32Flat, BvecFlat, IdsTuple, K, N)
            end;
        Got ->
            {error, {dim_mismatch, Dim, Got}}
    end.

-spec normalize(vector()) -> {ok, [float()]} | {error, term()}.
normalize(Vec) when is_list(Vec)   -> sied:l2_normalize_f32(Vec);
normalize(Vec) when is_binary(Vec) -> sied:l2_normalize_f32(f32_bin_to_list(Vec)).

-spec cosine_search(index(), Query :: vector(), K :: pos_integer()) ->
        {ok, [{id(), Score :: float()}]} | {error, term()}.
cosine_search(Ix, Query, K) when is_list(Query), is_integer(K), K > 0 ->
    case sied:l2_normalize_f32(Query) of
        {ok, NormQ} -> search(Ix, NormQ, K);
        Error       -> Error
    end;
cosine_search(Ix, Query, K) when is_binary(Query), is_integer(K), K > 0 ->
    cosine_search(Ix, f32_bin_to_list(Query), K).

-spec dump(index(), file:filename()) -> ok | {error, term()}.
%% @doc Serialise the index to a file.
%% Format: magic(4) + vsn(1) + dim(4) + n(4) + ids_len(4) + ids + f32_flat + bvec_flat.
dump(#{table := Tid, dim := Dim}, Path) ->
    [{?FLAT_KEY, F32Flat, BvecFlat, IdsTuple}] = ets:lookup(Tid, ?FLAT_KEY),
    N      = tuple_size(IdsTuple),
    IdsBin = term_to_binary(tuple_to_list(IdsTuple)),
    IdsLen = byte_size(IdsBin),
    Header = <<"KVEX", 1:8, Dim:32/big, N:32/big, IdsLen:32/big>>,
    file:write_file(Path, [Header, IdsBin, F32Flat, BvecFlat]).

-spec load(file:filename()) -> {ok, index()} | {error, term()}.
%% @doc Restore an index from a file produced by dump/2.
load(Path) ->
    case file:read_file(Path) of
        {ok, <<"KVEX", 1:8, Dim:32/big, N:32/big, IdsLen:32/big, Rest/binary>>} ->
            <<IdsBin:IdsLen/binary, Payload/binary>> = Rest,
            Ids      = binary_to_term(IdsBin, [safe]),
            F32Len   = Dim * 4,
            BVecLen  = (Dim + 7) div 8,
            F32Size  = N * F32Len,
            BVecSize = N * BVecLen,
            case Payload of
                <<F32Flat:F32Size/binary, BvecFlat:BVecSize/binary>> ->
                    Tid = ets:new(kvex, [set, protected]),
                    rebuild_from_flat(Tid, Ids, F32Flat, BvecFlat, F32Len, BVecLen, 0),
                    ets:insert(Tid, {?FLAT_KEY, F32Flat, BvecFlat, list_to_tuple(Ids)}),
                    {ok, #{table => Tid, dim => Dim}};
                _ ->
                    {error, bad_format}
            end;
        {ok, _} ->
            {error, bad_format};
        {error, _} = Err ->
            Err
    end.

%%%===================================================================
%%% Internal
%%%===================================================================

do_search(QBin, F32Flat, BvecFlat, IdsTuple, K, N) ->
    {ok, QQuantBin} = sied:to_binary_f32_bin(QBin),
    VecBLen   = byte_size(QQuantBin),
    VecF32Len = byte_size(QBin),
    CandCount = min(K * ?OVERSAMPLE, N),
    %% Phase 1 — SIMD POPCNT on flat binary, returns top-CandCount indices
    {ok, CandIdxs} = sied:hamming_topk_flat(QQuantBin, BvecFlat, VecBLen, CandCount),
    %% Phase 2 — SIMD dot-product on flat f32, returns [{Score, Idx}] sorted desc
    {ok, Scored} = sied:dot_product_topk_flat(QBin, F32Flat, VecF32Len, CandIdxs),
    {ok, [{element(Idx + 1, IdsTuple), Score}
          || {Score, Idx} <- lists:sublist(Scored, K)]}.

%% Rebuild flat cache from all records in the vec table.
%% Called after single add/3 — O(N) scan + binary concat.
rebuild_flat(Tid) ->
    All = ets:select(Tid, [{{'$1','$2','$3'}, [{'/=','$1',{const,?FLAT_KEY}}], [{{'$1','$2','$3'}}]}]),
    {F32Flat, BvecFlat, Ids} = lists:foldl(
        fun({Id, F32, BV}, {F, B, Is}) ->
            {<<F/binary, F32/binary>>, <<B/binary, BV/binary>>, [Id | Is]}
        end,
        {<<>>, <<>>, []},
        All
    ),
    ets:insert(Tid, {?FLAT_KEY, F32Flat, BvecFlat, list_to_tuple(lists:reverse(Ids))}).

build_entries([], _Dim, _Pos, EAcc, F32Acc, BvAcc, IdsAcc) ->
    {ok, lists:reverse(EAcc),
     iolist_to_binary(lists:reverse(F32Acc)),
     iolist_to_binary(lists:reverse(BvAcc)),
     lists:reverse(IdsAcc)};
build_entries([{Id, Vec0} | Rest], Dim, Pos, EAcc, F32Acc, BvAcc, IdsAcc) ->
    F32Bin = to_f32_bin(Vec0),
    case byte_size(F32Bin) div 4 of
        Dim ->
            {ok, BinVec} = sied:to_binary_f32_bin(F32Bin),
            build_entries(Rest, Dim, Pos + 1,
                [{Id, F32Bin, BinVec} | EAcc],
                [F32Bin  | F32Acc],
                [BinVec  | BvAcc],
                [Id      | IdsAcc]);
        Got ->
            {error, {dim_mismatch, Pos, Dim, Got}}
    end.

rebuild_from_flat(_Tid, [], _F32Flat, _BvecFlat, _F32Len, _BVecLen, _I) -> ok;
rebuild_from_flat(Tid, [Id | Rest], F32Flat, BvecFlat, F32Len, BVecLen, I) ->
    F32Off  = I * F32Len,
    BVOff   = I * BVecLen,
    <<_:F32Off/binary,  F32:F32Len/binary,  _/binary>> = F32Flat,
    <<_:BVOff/binary,   BV:BVecLen/binary,  _/binary>> = BvecFlat,
    ets:insert(Tid, {Id, F32, BV}),
    rebuild_from_flat(Tid, Rest, F32Flat, BvecFlat, F32Len, BVecLen, I + 1).

to_f32_bin(Vec) when is_binary(Vec) -> Vec;
to_f32_bin(Vec) when is_list(Vec)   ->
    << <<F:32/float-little>> || F <- Vec >>.

f32_bin_to_list(<<>>) -> [];
f32_bin_to_list(<<F:32/float-little, Rest/binary>>) ->
    [F | f32_bin_to_list(Rest)].
