# Class to hold original sentence with actual translated version.
class Sentence:
    def __init__(self, id_in, original_sentence_in, translated_version_in):
        self.id = id_in
        self.original = original_sentence_in
        self.translated = translated_version_in
