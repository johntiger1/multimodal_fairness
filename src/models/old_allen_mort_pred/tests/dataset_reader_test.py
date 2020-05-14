from allennlp.common.testing import AllenNlpTestCase

from allennlp.common.util import ensure_list
from allennlp.common.file_utils import cached_path

import sys

from src.models.allen_mort_pred.text_mortality.dataset_readers.SemanticScholarDatasetReader import SemanticScholarDatasetReader
PATH_OFFSET = "src/models/allen_mort_pred/"
class TestPatientNoteReader(AllenNlpTestCase):
    def __init__(self):
        # cached_path("https://github.com/allenai/allennlp-as-a-library-example/blob/master/tests/fixtures/s2_papers.jsonl")
        pass

    def test_read_from_file(self):
        reader = SemanticScholarDatasetReader()
        instances = ensure_list(reader.read(PATH_OFFSET + "tests/fixtures/s2_papers.jsonl"))
        instance1 = {"title": ["Interferring", "Discourse", "Relations", "in", "Context"],
                     "abstract": ["We", "investigate", "various", "contextual", "effects"],
                     "venue": "ACL"}

        instance2 = {"title": ["GRASPER", ":", "A", "Permissive", "Planning", "Robot"],
                     "abstract": ["Execut", "ion", "of", "classical", "plans"],
                     "venue": "AI"}

        instance3 = {"title": ["Route", "Planning", "under", "Uncertainty", ":", "The", "Canadian",
                               "Traveller", "Problem"],
                     "abstract": ["The", "Canadian", "Traveller", "problem", "is"],
                     "venue": "AI"}

        assert len(instances) == 10
        fields = instances[0].fields
        assert [t.text for t in fields["title"].tokens] == instance1["title"]
        assert [t.text for t in fields["abstract"].tokens[:5]] == instance1["abstract"]
        assert fields["label"].label == instance1["venue"]
        fields = instances[1].fields
        assert [t.text for t in fields["title"].tokens] == instance2["title"]
        assert [t.text for t in fields["abstract"].tokens[:5]] == instance2["abstract"]
        assert fields["label"].label == instance2["venue"]
        fields = instances[2].fields
        assert [t.text for t in fields["title"].tokens] == instance3["title"]
        assert [t.text for t in fields["abstract"].tokens[:5]] == instance3["abstract"]
        assert fields["label"].label == instance3["venue"]

'''ghetto pytest runner'''
if __name__ == "__main__":
    test = TestPatientNoteReader()
    test.test_read_from_file()
    print("done")