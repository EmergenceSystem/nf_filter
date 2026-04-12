# nf_filter
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.md)

An [em_filter](https://hex.pm/packages/em_filter) agent that searches the [AFNOR boutique](https://www.boutique.afnor.org/) for French NF, NF EN, NF ISO and related standards and returns results as [Emergence](https://github.com/EmergenceSystem/em_disco) results.

## Query

Any NF standard reference or keyword accepted by the AFNOR boutique search.

| Input form | Example |
|---|---|
| Standard reference | `NF EN ISO 9001`, `NF C 15-100` |
| Keyword | `sécurité électrique`, `qualité` |
| Number only | `15-100`, `9001` |

| Field | Example |
|---|---|
| title | `NF EN ISO 9001 — Systèmes de management de la qualité` |
| resume | short description from the AFNOR page |
| url | `https://www.boutique.afnor.org/fr-fr/...` |
| source | `boutique.afnor.org` |

## Usage

**Via curl (direct to em_disco):**

```bash
# By reference
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"value": "NF C 15-100", "capabilities": ["nf"]}'

# By keyword
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"value": "installations électriques", "capabilities": ["afnor"]}'
```

**Via Erlang shell:**

```erlang
emquest_cli:query(<<"NF EN ISO 9001">>).
emquest_cli:query(<<"normes bâtiment">>).
```

## Installation

```bash
git clone https://github.com/EmergenceSystem/nf_filter.git
cd nf_filter
rebar3 shell --apps nf_filter
```

Requires `em_disco` running on `localhost:8080` (configured in `emergence.conf`).

## Capabilities

`search`, `query`, `normes`, `nf`, `afnor`, `standards`, `reglementation`, `certification`

## License

Apache 2.0 — see [LICENSE.md](LICENSE.md).
