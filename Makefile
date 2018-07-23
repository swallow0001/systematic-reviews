DATA=$(PWD)/data
DATARIS=$(DATA)/ptsd_review/ris
DATACSV=$(DATA)/ptsd_review/csv
WORD2VEC=$(PWD)/word2vec

build : $(DATACSV)/schoot-lgmm-ptsd-initial.csv \
        $(DATACSV)/schoot-lgmm-ptsd-included-1.csv \
        $(DATACSV)/schoot-lgmm-ptsd-included-2.csv

download : $(WORD2VEC)/wiki.en.vec

$(WORD2VEC)/wiki.en.vec : 
	curl https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec -o $(WORD2VEC)/wiki.en.vec

$(DATACSV)/schoot-lgmm-ptsd-initial.csv : $(DATARIS)/schoot-lgmm-ptsd-initial.ris
	python python/risparser.py $(DATARIS)/schoot-lgmm-ptsd-initial.ris $(DATACSV)/schoot-lgmm-ptsd-initial.csv

$(DATACSV)/schoot-lgmm-ptsd-included-1.csv : $(DATARIS)/schoot-lgmm-ptsd-included-1.ris
	python python/risparser.py $(DATARIS)/schoot-lgmm-ptsd-included-1.ris $(DATACSV)/schoot-lgmm-ptsd-included-1.csv

$(DATACSV)/schoot-lgmm-ptsd-included-2.csv : $(DATARIS)/schoot-lgmm-ptsd-included-2.ris
	python python/risparser.py $(DATARIS)/schoot-lgmm-ptsd-included-2.ris $(DATACSV)/schoot-lgmm-ptsd-included-2.csv

clean:
	rm -rf $(DATACSV)
