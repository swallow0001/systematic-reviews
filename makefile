DATARIS=$(PWD)/data/example_dataset_1/ris
DATACSV=$(PWD)/data/example_dataset_1/csv

build : $(DATACSV)/01.\ Initial\ Search.csv \
        $(DATACSV)/02.\ included\ after\ title\ screening.csv \
        $(DATACSV)/03.\ finally\ included.csv

$(DATACSV)/01.\ Initial\ Search.csv : $(DATARIS)/01.\ Initial\ Search.txt
	python python/risparser.py $(DATARIS)/01.\ Initial\ Search.txt $(DATACSV)/01.\ Initial\ Search.csv

$(DATACSV)/02.\ included\ after\ title\ screening.csv : $(DATARIS)/02.\ included\ after\ title\ screening.txt
	python python/risparser.py $(DATARIS)/02.\ included\ after\ title\ screening.txt $(DATACSV)/02.\ included\ after\ title\ screening.csv

$(DATACSV)/03.\ finally\ included.csv : $(DATARIS)/03.\ finally\ included.txt
	python python/risparser.py $(DATARIS)/03.\ finally\ included.txt $(DATACSV)/03.\ finally\ included.csv

clean:
	rm -rf $(DATACSV)
