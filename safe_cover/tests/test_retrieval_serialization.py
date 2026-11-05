import numpy as np

from trident.retrieval import DenseRetriever


def test_dense_index_round_trip_uses_safe_numpy_archive(tmp_path) -> None:
    retriever = DenseRetriever.__new__(DenseRetriever)
    retriever.corpus = ["first passage", "second passage"]
    retriever.corpus_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    index_path = tmp_path / "index.npz"

    retriever.save_index(index_path)

    loaded = DenseRetriever.__new__(DenseRetriever)
    loaded.load_index(index_path)
    assert loaded.corpus == retriever.corpus
    assert loaded.corpus_embeddings is not None
    np.testing.assert_array_equal(loaded.corpus_embeddings, retriever.corpus_embeddings)
