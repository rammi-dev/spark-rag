"""Unit tests for GitHub Issues chunker."""

from spark_rag.chunking.issue_chunker import chunk_issue, _extract_pr_refs


SAMPLE_ISSUE = {
    "number": 48123,
    "title": "NullPointerException in Catalyst optimizer on Spark 4.1",
    "body": "When running a complex join query, the optimizer throws NPE.\n\n"
            "See also #47500 and https://github.com/apache/spark/pull/47501",
    "state": "closed",
    "user": {"login": "contributor123"},
    "labels": [{"name": "bug"}, {"name": "SQL"}, {"name": "priority:critical"}],
    "milestone": {"title": "4.1.1"},
    "created_at": "2025-06-15T10:00:00Z",
    "closed_at": "2025-07-01T14:30:00Z",
    "comments_data": [
        {
            "body": "I can reproduce this on Spark 4.1.0. Fix in #48124.",
            "user": {"login": "reviewer456"},
            "created_at": "2025-06-16T08:00:00Z",
        },
        {
            "body": "Fixed by GH-48124, merged.",
            "user": {"login": "committer789"},
            "created_at": "2025-07-01T14:00:00Z",
        },
    ],
}


class TestIssueChunker:
    def test_produces_body_and_comments(self):
        chunks = chunk_issue(SAMPLE_ISSUE)
        assert len(chunks) == 3  # 1 body + 2 comments
        body = [c for c in chunks if not c.is_comment]
        comments = [c for c in chunks if c.is_comment]
        assert len(body) == 1
        assert len(comments) == 2

    def test_issue_body_has_title(self):
        chunks = chunk_issue(SAMPLE_ISSUE)
        body = [c for c in chunks if not c.is_comment][0]
        assert "NullPointerException" in body.content
        assert "Catalyst optimizer" in body.content

    def test_issue_metadata(self):
        chunks = chunk_issue(SAMPLE_ISSUE)
        body = [c for c in chunks if not c.is_comment][0]
        assert body.issue_number == 48123
        assert body.state == "closed"
        assert body.author == "contributor123"
        assert "bug" in body.labels["labels"]
        assert "SQL" in body.labels["labels"]
        assert body.created_at == "2025-06-15T10:00:00Z"
        assert body.closed_at == "2025-07-01T14:30:00Z"

    def test_extracts_versions_from_labels_and_milestone(self):
        chunks = chunk_issue(SAMPLE_ISSUE)
        body = [c for c in chunks if not c.is_comment][0]
        versions = body.spark_versions_mentioned["versions"]
        assert "4.1.1" in versions  # from milestone

    def test_extracts_versions_from_content(self):
        chunks = chunk_issue(SAMPLE_ISSUE)
        body = [c for c in chunks if not c.is_comment][0]
        versions = body.spark_versions_mentioned["versions"]
        assert "4.1" in versions or "4.1.1" in versions

    def test_extracts_pr_refs(self):
        chunks = chunk_issue(SAMPLE_ISSUE)
        body = [c for c in chunks if not c.is_comment][0]
        prs = body.linked_prs["prs"]
        assert 47500 in prs
        assert 47501 in prs

    def test_comment_linked_to_parent(self):
        chunks = chunk_issue(SAMPLE_ISSUE)
        comments = [c for c in chunks if c.is_comment]
        for c in comments:
            assert c.parent_issue_number == 48123
            assert c.is_comment is True

    def test_comment_extracts_prs(self):
        chunks = chunk_issue(SAMPLE_ISSUE)
        comments = [c for c in chunks if c.is_comment]
        first_comment = comments[0]
        assert 48124 in first_comment.linked_prs["prs"]

    def test_comment_author(self):
        chunks = chunk_issue(SAMPLE_ISSUE)
        comments = [c for c in chunks if c.is_comment]
        assert comments[0].author == "reviewer456"
        assert comments[1].author == "committer789"

    def test_to_milvus_data(self):
        chunks = chunk_issue(SAMPLE_ISSUE)
        data = chunks[0].to_milvus_data([0.1] * 768)
        assert data["issue_number"] == 48123
        assert data["state"] == "closed"
        assert data["is_comment"] is False
        assert len(data["embedding"]) == 768

    def test_empty_body(self):
        issue = {
            "number": 1,
            "title": "Title only",
            "body": None,
            "state": "open",
            "user": {"login": "user"},
            "labels": [],
            "milestone": None,
            "created_at": "2025-01-01T00:00:00Z",
            "closed_at": None,
            "comments_data": [],
        }
        chunks = chunk_issue(issue)
        assert len(chunks) == 1
        assert "Title only" in chunks[0].content


class TestPRExtraction:
    def test_hash_format(self):
        assert 47500 in _extract_pr_refs("See #47500")

    def test_gh_format(self):
        assert 48124 in _extract_pr_refs("Fixed by GH-48124")

    def test_url_format(self):
        prs = _extract_pr_refs("https://github.com/apache/spark/pull/47501")
        assert 47501 in prs

    def test_multiple(self):
        prs = _extract_pr_refs("#47500 and #47501 and GH-47502")
        assert len(prs) == 3

    def test_no_short_numbers(self):
        prs = _extract_pr_refs("version #3 or #12")
        assert len(prs) == 0  # too short to be PR numbers
