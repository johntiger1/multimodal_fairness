
from allennlp.common.testing import ModelTestCase

class AcademicPaperClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('tests/fixtures/academic_paper_classifier.json',
                          'tests/fixtures/s2_papers.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)


if __name__ == "__main__":
    test = AcademicPaperClassifierTest()
    test.setUp()
    test.test_model_can_train_save_and_load()
    # print()