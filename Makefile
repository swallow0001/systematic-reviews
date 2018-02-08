DATA=$(PWD)/data
DATARIS=$(DATA)/example_dataset_1/ris
DATACSV=$(DATA)/example_dataset_1/csv

build : $(DATACSV)/schoot-lgmm-ptsd-initial.csv \
        $(DATACSV)/schoot-lgmm-ptsd-included-1.csv \
        $(DATACSV)/schoot-lgmm-ptsd-included-2.csv

download : $(DATA)/wiki.en.vec

$(DATA)/wiki.en.vec : 
	curl https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec -o $(DATA)/wiki.en.vec

$(DATACSV)/schoot-lgmm-ptsd-initial.csv : $(DATARIS)/schoot-lgmm-ptsd-initial.ris
	python python/risparser.py $(DATARIS)/schoot-lgmm-ptsd-initial.ris $(DATACSV)/schoot-lgmm-ptsd-initial.csv

$(DATACSV)/schoot-lgmm-ptsd-included-1.csv : $(DATARIS)/schoot-lgmm-ptsd-included-1.ris
	python python/risparser.py $(DATARIS)/schoot-lgmm-ptsd-included-1.ris $(DATACSV)/schoot-lgmm-ptsd-included-1.csv

$(DATACSV)/schoot-lgmm-ptsd-included-2.csv : $(DATARIS)/schoot-lgmm-ptsd-included-2.ris
	python python/risparser.py $(DATARIS)/schoot-lgmm-ptsd-included-2.ris $(DATACSV)/schoot-lgmm-ptsd-included-2.csv

clean:
	rm -rf $(DATACSV)
