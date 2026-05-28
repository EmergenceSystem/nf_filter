-module(nf_filter_test).
-include_lib("eunit/include/eunit.hrl").

%%====================================================================
%% is_norm_href/1
%%====================================================================

is_norm_href_norme_test() ->
    ?assert(nf_filter_app:is_norm_href("/fr-fr/norme/nf-iso-9001/qualite/fa123/1")).

is_norm_href_produit_test() ->
    ?assert(nf_filter_app:is_norm_href("/fr-fr/produit/nf-en-12345")).

is_norm_href_other_test() ->
    ?assertNot(nf_filter_app:is_norm_href("/fr-fr/accueil")),
    ?assertNot(nf_filter_app:is_norm_href("https://www.google.com")),
    ?assertNot(nf_filter_app:is_norm_href("/fr-fr/contact")).

%%====================================================================
%% build_embryo/2
%%====================================================================

build_embryo_relative_test() ->
    E = nf_filter_app:build_embryo("/fr-fr/norme/nf-iso-9001/fa1/1", "NF ISO 9001"),
    ?assertEqual(<<"url">>, maps:get(<<"type">>, E)),
    Props = maps:get(<<"properties">>, E),
    ?assertEqual(<<"https://www.boutique.afnor.org/fr-fr/norme/nf-iso-9001/fa1/1">>,
                 maps:get(<<"url">>,    Props)),
    ?assertEqual(<<"NF ISO 9001">>, maps:get(<<"title">>, Props)),
    ?assertEqual(<<"boutique.afnor.org">>, maps:get(<<"source">>, Props)).

build_embryo_absolute_test() ->
    E = nf_filter_app:build_embryo("https://www.boutique.afnor.org/fr-fr/norme/x", "NF X"),
    Props = maps:get(<<"properties">>, E),
    ?assertEqual(<<"https://www.boutique.afnor.org/fr-fr/norme/x">>,
                 maps:get(<<"url">>, Props)).

%%====================================================================
%% extract_norm_links/1
%%====================================================================

extract_norm_links_basic_test() ->
    Html = "<html><body>"
           "<a href=\"/fr-fr/norme/nf-iso-9001/qualite/fa123/1\">"
           "NF ISO 9001:2015 — Systèmes de management de la qualité</a>"
           "<a href=\"/fr-fr/norme/nf-en-12354/acoustique/fa456/2\">"
           "NF EN 12354 — Acoustique du bâtiment</a>"
           "<a href=\"/fr-fr/accueil\">Accueil</a>"
           "</body></html>",
    Results = nf_filter_app:extract_norm_links(Html),
    ?assertEqual(2, length(Results)).

extract_norm_links_dedup_test() ->
    Href = "/fr-fr/norme/nf-iso-9001/fa1/1",
    Html = "<a href=\"" ++ Href ++ "\">NF ISO 9001</a>"
           "<a href=\"" ++ Href ++ "\">NF ISO 9001</a>",
    Results = nf_filter_app:extract_norm_links(Html),
    ?assertEqual(1, length(Results)).

extract_norm_links_empty_test() ->
    ?assertEqual([], nf_filter_app:extract_norm_links("<html><body>no links</body></html>")).

extract_norm_links_no_norm_test() ->
    Html = "<a href=\"/fr-fr/accueil\">Accueil</a>"
           "<a href=\"https://example.com\">External</a>",
    ?assertEqual([], nf_filter_app:extract_norm_links(Html)).
