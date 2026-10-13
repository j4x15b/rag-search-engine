import streamlit as st
from lib.search_utils import load_movies
from lib.keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from lib.hybrid_search import HybridSearch
from lib.hybrid_search import llm_enhancement
from lib.cross_encoder_reranker import CrossEncoderReranker
from lib.hybrid_search import rrf_search_command

st.set_page_config(
    page_title="Movie Search",
    #page_icon="",
    layout="wide",
)
tab_suche, tab_filme = st.tabs(["Suche", "Filmarchiv"])


# --- Suchkomponenten einmalig initialisieren ---

@st.cache_resource
def get_documents():
    return load_movies()

@st.cache_resource
def get_keyword_search():
    index = InvertedIndex()

    cache_files = [
        index.index_path,
        index.docmap_path,
        index.term_frequencies_path,
        index.doc_lengths_path,
    ]

    if all(path.exists() for path in cache_files):
        index.load()
    else:
        index.build()
        index.save()

    return index


@st.cache_resource
def get_semantic_search():
    search = ChunkedSemanticSearch()
    search.load_or_create_chunk_embeddings(get_documents())
    return search


@st.cache_resource
def get_hybrid_search():
    return HybridSearch(get_documents())

@st.cache_resource
def get_cross_encoder_reranker():
    return CrossEncoderReranker()

# --- Eingabe ---
with tab_suche:
    
    st.title("Movie Search")
    st.write("Try out the different search methods.")

    with st.form("search_form"):
        query = st.text_input(
            "Search term",
            placeholder="e.g., science fiction or artificial intelligence, the boy who lives",
        )

        search_type = st.radio(
            "Search method",
            ["Keyword", "Semantic", "Hybrid"],
            horizontal=True,
        )

        limit = st.slider(
            "Number of results",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
        )


        if search_type == "Hybrid":
            hybrid_method = st.radio(
                "Hybrid method",
                ["weighted search", "RRF"],
                horizontal=True,
            )

            alpha = st.slider(
                "Alpha – weighting of semantic search",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="0 = keyword only, 0.5 = balanced, 1 = semantic search only",
            )

            k = st.slider(
                        "RRF K value (search parameter)",
                        min_value=1,
                        max_value=100,
                        value=60,
                        step=1,
                        help="Affects how strongly ranking positions influence the RRF score."
            )
        
        rerank_method = "none"

        if search_type == "Hybrid" and hybrid_method == "RRF":
            rerank_method = st.selectbox(
                "Reranking",
                options=["none", "individual", "batch", "cross_encoder"],
                format_func=lambda x: {
                    "none": "no reranking",
                    "individual": "LLM - individual movies",
                    "batch": "LLM - Batch",
                    "cross_encoder": "local cross-encoder",
                }[x],
            )

        else:
            hybrid_method = "Weighted Search"
            alpha = 0.5

        enhance_query = st.checkbox(
            "Enhanced Search",
            value=False,
            help="Refine the search query with an LLM before searching."
        )

        enhance_method = None

        if enhance_query:
            enhance_method = st.radio(
                "method",
                ["expand", "rewrite", "spell"],
                format_func=lambda x: {
                    "expand": "Expand - Expand the query",
                    "rewrite": "Rewrite - Rephrase the query",
                    "spell": "Spell - Correct the spelling",
                }[x],
                horizontal=True,
            )

        submitted = st.form_submit_button("search")


    # --- Suche und Ergebnisse ---

    
    if submitted:
        if not query.strip():
            st.warning("Please enter a search term.")
        else:
            search_query = query

            # 1. Query optional erweitern
            if enhance_query:
                with st.spinner("Expanding search query..."):
                    search_query = llm_enhancement(
                        query=query,
                        enhance_method=enhance_method,
                        model="openrouter",
                    )

                st.write("Expanded query:", search_query)

            # 2. Suche mit der Original- oder erweiterten Query ausführen
            try:
                with st.spinner("Searching..."):

                    if search_type == "Keyword":
                        index = get_keyword_search()
                        raw_results = index.bm25_search(
                            search_query,
                            limit=limit,
                        )

                        results = []
                        for doc_id, score in raw_results:
                            doc = index.docmap[doc_id]
                            results.append({
                                "title": doc["title"],
                                "description": doc["description"],
                                "score": float(score),
                            })

                    elif search_type == "Semantic":
                        search = get_semantic_search()
                        results = search.search_chunks(
                            search_query,
                            limit=limit,
                        )

                    else:
                        search = get_hybrid_search()

                        if hybrid_method == "weighted search":
                            results = search.weighted_search(
                                query=search_query,
                                alpha=alpha,
                                limit=limit,
                            )
                        else:
                            reranker = None

                            if rerank_method == "cross_encoder":
                                reranker = get_cross_encoder_reranker()

                            results = rrf_search_command(
                                query=search_query,
                                k=k,
                                limit=limit,
                                enhance_method=None,
                                rerank_method=rerank_method,
                                reranker=reranker,
                            )
                            if rerank_method in ("individual", "batch", "cross_encoder"):
                                results = [
                                    {**result, "rerank_rank": rank}
                                    for rank, result in results
                                ]

                # 3. Ergebnisse wie bisher anzeigen
                st.subheader(f"Results ({len(results)})")

                if not results:
                    st.info("No results found.")

                for i, result in enumerate(results, start=1):
                    title = result.get("title", "untitled")
                    description = result.get("description", "")

                    score = result.get(
                        "hybrid_score",
                        result.get(
                            "rrf_score",
                            result.get("score", result.get("bm25_score")),
                        ),
                    )

                    heading = f"{i}. {title}"
                    if score is not None:
                        heading += f" — Score: {score:.3f}"

                    with st.expander(heading):
                        preview = description[:1000]
                        if len(description) > 1000:
                            preview += " …"

                        st.write(preview)

                        with st.expander("show full description"):
                            st.write(description)

                        score_columns = [
                            ("hybrid_score", "Hybrid"),
                            ("semantic_score", "Semantic"),
                            ("bm25_score", "BM25"),
                            ("rrf_score", "RRF"),
                            ("bm_25_rank", "Keyword-Rank"),
                            ("semantic_rank", "Semantic-Rank"),
                            ("rerank_score", "Cross-Encoder"),
                            #("rerank_rank", "Rerank-Rank")
                        ]

                        available_scores = [
                            (key, label, result[key])
                            for key, label in score_columns
                            if key in result
                            and isinstance(result[key], (int, float))
                        ]

                        if available_scores:
                            columns = st.columns(len(available_scores))

                            for column, (key, label, value) in zip(
                                columns, available_scores
                            ):
                                if key in ("bm_25_rank", "semantic_rank"):
                                    display_value = str(int(value))
                                else:
                                    display_value = f"{value:.4f}"

                                column.metric(label, display_value)

                        with st.expander("show raw data"):
                            st.json(result)

            except Exception as exc:
                st.error(f"An error occurred while searching: {exc}")

with tab_filme:
    documents = get_documents()
    st.subheader(f"film library ({len(documents)} films)")

    page_size = 100
    total_pages = max(1, (len(documents) + page_size - 1) // page_size)

    # Aktuelle Seite im Session State speichern
    if "film_library_page" not in st.session_state:
        st.session_state.filmbibliothek_seite = 1

    # Sicherstellen, dass die Seite gültig bleibt
    page = min(st.session_state.filmbibliothek_seite, total_pages)
    st.session_state.film_library_page = page

    # Navigation
    col_prev, col_page, col_next = st.columns([1, 2, 1])

    with col_page:
        new_page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=page,
            step=1,
            label_visibility="collapsed",
            key="film library page input",
        )

        st.caption(f"von {total_pages} Seiten")

    
        if new_page != page:
            st.session_state.filmbibliothek_seite = int(new_page)
            st.rerun()
    start = (page - 1) * page_size
    end = min(start + page_size, len(documents))

    st.caption(f"Movies {start + 1}-{end} of {len(documents)}")
    st.divider()

    for movie in documents[start:end]:
        title = movie.get("title", "untitled")
        movie_id = movie.get("id", "no ID")
        description = movie.get("description", "")

        with st.expander(f"{movie_id} - {title}"):
            st.write(description)
