%%%-------------------------------------------------------------------
%%% @doc French NF standards search agent (AFNOR boutique).
%%%
%%% Searches the AFNOR boutique catalogue for NF, NF EN, NF ISO and
%%% related standards.  Returns url embryos linking to the norm page.
%%%
%%% Handler contract: handle/2 (Body, Memory) -> {RawList, Memory}.
%%% @end
%%%-------------------------------------------------------------------
-module(nf_filter_app).
-behaviour(application).

-export([start/2, stop/1]).
-export([handle/2, base_capabilities/0]).

%% Exported for testing
-export([extract_norm_links/1, is_norm_href/1, build_embryo/2]).

-define(SEARCH_URL, "https://www.boutique.afnor.org/fr-fr/recherche?q=").
-define(BASE_URL,   "https://www.boutique.afnor.org").
-define(USER_AGENT, "Emergence-NF-Filter/1.0").

%%====================================================================
%% Capability cascade
%%====================================================================

-spec base_capabilities() -> [binary()].
base_capabilities() ->
    em_filter:base_capabilities() ++
        [<<"normes">>, <<"nf">>, <<"afnor">>, <<"standards">>,
         <<"reglementation">>, <<"certification">>].

%%====================================================================
%% Application lifecycle
%%====================================================================

start(_Type, _Args) ->
    case nf_filter_sup:start_link() of
        {ok, Pid} ->
            ok = start_pop_and_http(),
            {ok, Pid};
        Error ->
            Error
    end.

stop(_State) ->
    catch cowboy:stop_listener(nf_filter_query_listener),
    catch em_pop_sup:stop_node(nf_filter),
    ok.

%%====================================================================
%% Internal
%%====================================================================

start_pop_and_http() ->
    PopPort   = application:get_env(nf_filter, pop_port,   9472),
    QueryPort = application:get_env(nf_filter, query_port, 9473),
    Seeds     = application:get_env(nf_filter, pop_seeds,  []),
    Vec = em_filter_vec:from_capabilities(base_capabilities()),
    catch em_pop_sup:stop_node(nf_filter),
    catch cowboy:stop_listener(nf_filter_query_listener),
    {ok, PopPid} = em_pop_sup:start_node(nf_filter, #{
        port            => PopPort,
        query_port      => QueryPort,
        vector          => Vec,
        max_peers       => 100,
        gossip_interval => 5_000
    }),
    lists:foreach(
        fun({H, P}) -> catch em_pop_node:add_peer(PopPid, H, P) end,
        Seeds),
    Dispatch = cowboy_router:compile([
        {'_', [{"/agent/query", em_filter_http,
                #{server => nf_filter_server}}]}
    ]),
    {ok, _} = cowboy:start_clear(nf_filter_query_listener,
                                  [{port, QueryPort}],
                                  #{env => #{dispatch => Dispatch}}),
    logger:notice("[nf_filter] gossip port ~w  query port ~w",
                  [PopPort, QueryPort]),
    ok.

handle(Body, Memory) when is_binary(Body) ->
    {search(Body), Memory};
handle(_Body, Memory) ->
    {[], Memory}.

%%====================================================================
%% Search
%%====================================================================

search(QueryBin) ->
    {Query, Timeout} = extract_params(QueryBin),
    Url = ?SEARCH_URL ++ uri_string:quote(Query),
    Headers = [{"User-Agent", ?USER_AGENT},
               {"Accept",     "text/html,application/xhtml+xml"}],
    case httpc:request(get, {Url, Headers},
                       [{timeout, Timeout * 1000}],
                       [{body_format, binary}]) of
        {ok, {{_, 200, _}, _, Body}} ->
            Html = binary_to_list(Body),
            extract_norm_links(Html);
        _ ->
            []
    end.

extract_params(JsonBinary) ->
    try json:decode(JsonBinary) of
        Map when is_map(Map) ->
            Value   = binary_to_list(maps:get(<<"value">>, Map,
                          maps:get(<<"query">>, Map, <<"">>))),
            Timeout = case maps:get(<<"timeout">>, Map, undefined) of
                undefined            -> 10;
                T when is_integer(T) -> T;
                T when is_binary(T)  -> binary_to_integer(T)
            end,
            {Value, Timeout};
        _ ->
            {binary_to_list(JsonBinary), 10}
    catch
        _:_ -> {binary_to_list(JsonBinary), 10}
    end.

%%====================================================================
%% HTML parsing
%%====================================================================

%% @doc Extract NF norm links from AFNOR boutique search HTML.
%% Regex-based extraction to handle any page structure.
-spec extract_norm_links(string() | binary()) -> [map()].
extract_norm_links(Html) ->
    Bin     = unicode:characters_to_binary(Html),
    Pattern = "<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
    case re:run(Bin, Pattern,
                [global, {capture, all_but_first, binary}, caseless, dotall, unicode]) of
        {match, Pairs} ->
            Seen = sets:new(),
            {Results, _} = lists:foldl(fun([Href, Inner], {Acc, Visited}) ->
                Text = string:trim(em_filter:get_text(binary_to_list(Inner))),
                case is_norm_href(binary_to_list(Href))
                     andalso Text =/= ""
                     andalso not sets:is_element(Href, Visited) of
                    true ->
                        Embryo = build_embryo(binary_to_list(Href), Text),
                        {[Embryo | Acc], sets:add_element(Href, Visited)};
                    false ->
                        {Acc, Visited}
                end
            end, {[], Seen}, Pairs),
            lists:reverse(Results);
        _ ->
            []
    end.

%% @doc True if the href points to an AFNOR norm page.
-spec is_norm_href(string()) -> boolean().
is_norm_href(Href) ->
    re:run(Href, "/fr-fr/norme/", [{capture, none}]) =:= match
    orelse re:run(Href, "/fr-fr/produit/", [{capture, none}]) =:= match.

%% @doc Build a url embryo from an AFNOR norm href and its link text.
-spec build_embryo(string(), string()) -> map().
build_embryo(Href, Title) ->
    FullUrl = case string:prefix(Href, "http") of
        nomatch -> ?BASE_URL ++ Href;
        _       -> Href
    end,
    #{<<"type">>       => <<"url">>,
      <<"properties">> => #{
          <<"url">>    => unicode:characters_to_binary(FullUrl),
          <<"title">>  => unicode:characters_to_binary(Title),
          <<"source">> => <<"boutique.afnor.org">>
      }}.
