%%%-------------------------------------------------------------------
%%% @doc sied - SIMD operations for Erlang
%%%
%%% High-performance vectorized operations using SIMD instructions
%%% via Rust NIF with simdeez. Provides runtime SIMD detection and
%%% automatic dispatch to SSE2, SSE4.1, AVX2, or NEON instructions.
%%%
%%% @end
%%%-------------------------------------------------------------------
-module(sied).

%% Basic Arithmetic Operations
-export([
    add_f32/2,
    add_f64/2,
    subtract_f32/2,
    subtract_f64/2,
    multiply_f32/2,
    multiply_f64/2,
    divide_f32/2,
    divide_f64/2
]).

%% Reduction Operations
-export([
    dot_product_f32/2,
    dot_product_f64/2,
    sum_f32/1,
    sum_f64/1
]).

%% Statistical Operations
-export([
    mean_f32/1,
    mean_f64/1,
    variance_f32/1,
    variance_f64/1,
    std_dev_f32/1,
    std_dev_f64/1
]).

%% Min/Max Operations
-export([
    min_f32/1,
    min_f64/1,
    max_f32/1,
    max_f64/1,
    min_elementwise_f32/2,
    min_elementwise_f64/2,
    max_elementwise_f32/2,
    max_elementwise_f64/2
]).

%% Unary Operations
-export([
    abs_f32/1,
    abs_f64/1,
    sqrt_f32/1,
    sqrt_f64/1,
    negate_f32/1,
    negate_f64/1
]).

%% Batch Operations (vector search)
-export([
    dot_product_batch_f32/2,
    dot_product_batch_f32_bin/2,
    dot_product_batch_f64/2
]).

%% Vector Norm and Normalization
-export([
    l2_norm_f32/1,
    l2_norm_f64/1,
    l2_normalize_f32/1,
    l2_normalize_f64/1,
    l2_normalize_batch_f32/1,
    l2_normalize_batch_f64/1
]).

%% Cosine Similarity
-export([
    cosine_similarity_f32/2,
    cosine_similarity_f64/2,
    cosine_similarity_batch_f32/2,
    cosine_similarity_batch_f64/2
]).

%% Binary Quantization
-export([
    to_binary_f32/1,
    to_binary_f32_bin/1,
    hamming_distance_batch/2,
    hamming_topk_flat/4,
    dot_product_topk_flat/4
]).

-on_load(init/0).

-define(APPNAME, sied).
-define(LIBNAME, sied).

%%%===================================================================
%%% NIF Loading
%%%===================================================================

%% @private
%% @doc Initialize and load the NIF library
init() ->
    SoName = case code:priv_dir(?APPNAME) of
        {error, bad_name} ->
            case filelib:is_dir(filename:join(["..", priv])) of
                true ->
                    filename:join(["..", priv, ?LIBNAME]);
                false ->
                    filename:join([priv, ?LIBNAME])
            end;
        Dir ->
            filename:join(Dir, ?LIBNAME)
    end,
    erlang:load_nif(SoName, 0).

%%%===================================================================
%%% Basic Arithmetic Operations
%%%===================================================================

%% @doc Element-wise addition of two f32 vectors
%% @param A First list of floats
%% @param B Second list of floats (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec add_f32([float()], [float()]) -> {ok, [float()]} | {error, term()}.
add_f32(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise addition of two f64 vectors
%% @param A First list of doubles
%% @param B Second list of doubles (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec add_f64([float()], [float()]) -> {ok, [float()]} | {error, term()}.
add_f64(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise subtraction of two f32 vectors
%% @param A First list of floats
%% @param B Second list of floats (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec subtract_f32([float()], [float()]) -> {ok, [float()]} | {error, term()}.
subtract_f32(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise subtraction of two f64 vectors
%% @param A First list of doubles
%% @param B Second list of doubles (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec subtract_f64([float()], [float()]) -> {ok, [float()]} | {error, term()}.
subtract_f64(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise multiplication of two f32 vectors
%% @param A First list of floats
%% @param B Second list of floats (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec multiply_f32([float()], [float()]) -> {ok, [float()]} | {error, term()}.
multiply_f32(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise multiplication of two f64 vectors
%% @param A First list of doubles
%% @param B Second list of doubles (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec multiply_f64([float()], [float()]) -> {ok, [float()]} | {error, term()}.
multiply_f64(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise division of two f32 vectors
%% @param A First list of floats (numerators)
%% @param B Second list of floats (denominators, must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec divide_f32([float()], [float()]) -> {ok, [float()]} | {error, term()}.
divide_f32(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise division of two f64 vectors
%% @param A First list of doubles (numerators)
%% @param B Second list of doubles (denominators, must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec divide_f64([float()], [float()]) -> {ok, [float()]} | {error, term()}.
divide_f64(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%%%===================================================================
%%% Reduction Operations
%%%===================================================================

%% @doc Compute dot product of two f32 vectors
%% Computes the scalar product: sum(A[i] * B[i])
%% @param A First vector
%% @param B Second vector (must be same length)
%% @returns {ok, Scalar} | {error, Reason}
-spec dot_product_f32([float()], [float()]) -> {ok, float()} | {error, term()}.
dot_product_f32(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute dot product of two f64 vectors
%% Computes the scalar product: sum(A[i] * B[i])
%% @param A First vector
%% @param B Second vector (must be same length)
%% @returns {ok, Scalar} | {error, Reason}
-spec dot_product_f64([float()], [float()]) -> {ok, float()} | {error, term()}.
dot_product_f64(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute sum of all elements in an f32 vector
%% @param A List of floats to sum
%% @returns {ok, Sum} | {error, Reason}
-spec sum_f32([float()]) -> {ok, float()} | {error, term()}.
sum_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute sum of all elements in an f64 vector
%% @param A List of doubles to sum
%% @returns {ok, Sum} | {error, Reason}
-spec sum_f64([float()]) -> {ok, float()} | {error, term()}.
sum_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%%%===================================================================
%%% Statistical Operations
%%%===================================================================

%% @doc Compute arithmetic mean of an f32 vector
%% @param A List of floats
%% @returns {ok, Mean} | {error, Reason}
-spec mean_f32([float()]) -> {ok, float()} | {error, term()}.
mean_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute arithmetic mean of an f64 vector
%% @param A List of doubles
%% @returns {ok, Mean} | {error, Reason}
-spec mean_f64([float()]) -> {ok, float()} | {error, term()}.
mean_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute variance of an f32 vector
%% @param A List of floats
%% @returns {ok, Variance} | {error, Reason}
-spec variance_f32([float()]) -> {ok, float()} | {error, term()}.
variance_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute variance of an f64 vector
%% @param A List of doubles
%% @returns {ok, Variance} | {error, Reason}
-spec variance_f64([float()]) -> {ok, float()} | {error, term()}.
variance_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute standard deviation of an f32 vector
%% @param A List of floats
%% @returns {ok, StdDev} | {error, Reason}
-spec std_dev_f32([float()]) -> {ok, float()} | {error, term()}.
std_dev_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute standard deviation of an f64 vector
%% @param A List of doubles
%% @returns {ok, StdDev} | {error, Reason}
-spec std_dev_f64([float()]) -> {ok, float()} | {error, term()}.
std_dev_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%%%===================================================================
%%% Min/Max Operations
%%%===================================================================

%% @doc Find minimum value in an f32 vector
%% @param A List of floats
%% @returns {ok, Min} | {error, Reason}
-spec min_f32([float()]) -> {ok, float()} | {error, term()}.
min_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Find minimum value in an f64 vector
%% @param A List of doubles
%% @returns {ok, Min} | {error, Reason}
-spec min_f64([float()]) -> {ok, float()} | {error, term()}.
min_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Find maximum value in an f32 vector
%% @param A List of floats
%% @returns {ok, Max} | {error, Reason}
-spec max_f32([float()]) -> {ok, float()} | {error, term()}.
max_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Find maximum value in an f64 vector
%% @param A List of doubles
%% @returns {ok, Max} | {error, Reason}
-spec max_f64([float()]) -> {ok, float()} | {error, term()}.
max_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise minimum of two f32 vectors
%% @param A First list of floats
%% @param B Second list of floats (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec min_elementwise_f32([float()], [float()]) -> {ok, [float()]} | {error, term()}.
min_elementwise_f32(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise minimum of two f64 vectors
%% @param A First list of doubles
%% @param B Second list of doubles (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec min_elementwise_f64([float()], [float()]) -> {ok, [float()]} | {error, term()}.
min_elementwise_f64(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise maximum of two f32 vectors
%% @param A First list of floats
%% @param B Second list of floats (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec max_elementwise_f32([float()], [float()]) -> {ok, [float()]} | {error, term()}.
max_elementwise_f32(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Element-wise maximum of two f64 vectors
%% @param A First list of doubles
%% @param B Second list of doubles (must be same length)
%% @returns {ok, Result} | {error, Reason}
-spec max_elementwise_f64([float()], [float()]) -> {ok, [float()]} | {error, term()}.
max_elementwise_f64(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%%%===================================================================
%%% Unary Operations
%%%===================================================================

%% @doc Compute absolute value of an f32 vector
%% @param A List of floats
%% @returns {ok, Result} | {error, Reason}
-spec abs_f32([float()]) -> {ok, [float()]} | {error, term()}.
abs_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute absolute value of an f64 vector
%% @param A List of doubles
%% @returns {ok, Result} | {error, Reason}
-spec abs_f64([float()]) -> {ok, [float()]} | {error, term()}.
abs_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute square root of an f32 vector
%% @param A List of floats (must be non-negative)
%% @returns {ok, Result} | {error, Reason}
-spec sqrt_f32([float()]) -> {ok, [float()]} | {error, term()}.
sqrt_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute square root of an f64 vector
%% @param A List of doubles (must be non-negative)
%% @returns {ok, Result} | {error, Reason}
-spec sqrt_f64([float()]) -> {ok, [float()]} | {error, term()}.
sqrt_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Negate an f32 vector (multiply by -1)
%% @param A List of floats
%% @returns {ok, Result} | {error, Reason}
-spec negate_f32([float()]) -> {ok, [float()]} | {error, term()}.
negate_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Negate an f64 vector (multiply by -1)
%% @param A List of doubles
%% @returns {ok, Result} | {error, Reason}
-spec negate_f64([float()]) -> {ok, [float()]} | {error, term()}.
negate_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%%%===================================================================
%%% Batch Operations (vector search)
%%%===================================================================

%% @doc Compute dot product of Query against every vector in Vecs.
%% Returns {ok, [Score]} in input order. One NIF call for the whole batch.
-spec dot_product_batch_f32([float()], [[float()]]) -> {ok, [float()]} | {error, term()}.
dot_product_batch_f32(_Query, _Vecs) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Dot product of a query f32 binary against a list of f32 binaries.
%% Binaries are little-endian IEEE 754 f32 (4 bytes per element).
%% Avoids Erlang float-list marshalling — use with kvex f32-binary storage.
-spec dot_product_batch_f32_bin(binary(), [binary()]) -> {ok, [float()]} | {error, term()}.
dot_product_batch_f32_bin(_Query, _Vecs) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Compute dot product of Query against every f64 vector in Vecs.
-spec dot_product_batch_f64([float()], [[float()]]) -> {ok, [float()]} | {error, term()}.
dot_product_batch_f64(_Query, _Vecs) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%%%===================================================================
%%% Vector Norm and Normalization
%%%===================================================================

%% @doc L2 (Euclidean) norm of an f32 vector
-spec l2_norm_f32([float()]) -> {ok, float()} | {error, term()}.
l2_norm_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc L2 (Euclidean) norm of an f64 vector
-spec l2_norm_f64([float()]) -> {ok, float()} | {error, term()}.
l2_norm_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc L2-normalize an f32 vector to unit length.
%% Returns the original vector if its norm is zero.
-spec l2_normalize_f32([float()]) -> {ok, [float()]} | {error, term()}.
l2_normalize_f32(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc L2-normalize an f64 vector to unit length.
-spec l2_normalize_f64([float()]) -> {ok, [float()]} | {error, term()}.
l2_normalize_f64(_A) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc L2-normalize a batch of f32 vectors.
-spec l2_normalize_batch_f32([[float()]]) -> {ok, [[float()]]} | {error, term()}.
l2_normalize_batch_f32(_Vecs) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc L2-normalize a batch of f64 vectors.
-spec l2_normalize_batch_f64([[float()]]) -> {ok, [[float()]]} | {error, term()}.
l2_normalize_batch_f64(_Vecs) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%%%===================================================================
%%% Cosine Similarity
%%%===================================================================

%% @doc Cosine similarity between two f32 vectors: dot(A,B) / (|A| * |B|).
%% Returns a value in [-1.0, 1.0].
-spec cosine_similarity_f32([float()], [float()]) -> {ok, float()} | {error, term()}.
cosine_similarity_f32(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Cosine similarity between two f64 vectors.
-spec cosine_similarity_f64([float()], [float()]) -> {ok, float()} | {error, term()}.
cosine_similarity_f64(_A, _B) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Batch cosine similarity: one f32 query against many f32 vectors.
-spec cosine_similarity_batch_f32([float()], [[float()]]) -> {ok, [float()]} | {error, term()}.
cosine_similarity_batch_f32(_Query, _Vecs) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Batch cosine similarity: one f64 query against many f64 vectors.
-spec cosine_similarity_batch_f64([float()], [[float()]]) -> {ok, [float()]} | {error, term()}.
cosine_similarity_batch_f64(_Query, _Vecs) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%%%===================================================================
%%% Binary Quantization
%%%===================================================================

%% @doc 1-bit quantize an f32 vector. Each element becomes 1 if above mean,
%% else 0. Returns a packed binary: 128 dims → 16 bytes.
%% Use hamming_distance_batch/2 to search over quantized vectors.
-spec to_binary_f32([float()]) -> {ok, binary()} | {error, term()}.
to_binary_f32(_Vec) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Like to_binary_f32/1 but accepts a little-endian f32 binary instead of a float list.
%% Zero-copy path when the vector is already stored as a binary (e.g. in kvex ETS).
-spec to_binary_f32_bin(binary()) -> {ok, binary()} | {error, term()}.
to_binary_f32_bin(_Data) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Batch hamming distance between a query binary and a list of binaries.
%% Returns {ok, [Distance]} where lower distance means more similar.
%% Uses u64 POPCNT for speed. Intended for the first phase of two-phase
%% ANN search: hamming_distance_batch → filter top-N → dot_product_batch.
-spec hamming_distance_batch(binary(), [binary()]) -> {ok, [non_neg_integer()]} | {error, term()}.
hamming_distance_batch(_Query, _Vecs) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Hamming top-K from a flat binary buffer (all vectors concatenated).
%% Returns {ok, [Idx]} — the indices of the top_k closest vectors, sorted ascending
%% by Hamming distance. O(N) partition + O(K log K) sort, no per-element Erlang overhead.
-spec hamming_topk_flat(binary(), binary(), pos_integer(), pos_integer()) ->
        {ok, [non_neg_integer()]} | {error, term()}.
hamming_topk_flat(_Query, _FlatVecs, _VecLen, _TopK) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).

%% @doc Dot-product scoring of selected vectors from a flat f32 binary.
%% indices: list produced by hamming_topk_flat.
%% Returns {ok, [{Score, Idx}]} sorted by descending score.
-spec dot_product_topk_flat(binary(), binary(), pos_integer(), [non_neg_integer()]) ->
        {ok, [{float(), non_neg_integer()}]} | {error, term()}.
dot_product_topk_flat(_Query, _FlatF32, _VecByteLen, _Indices) ->
    erlang:nif_error({not_loaded, [{module, ?MODULE}, {line, ?LINE}]}).
