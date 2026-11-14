"""Step 1: download NFCorpus, embed it, build the member / non-member split. Run once."""
from ragmia.data import prepare_data

if __name__ == "__main__":
    prepare_data()