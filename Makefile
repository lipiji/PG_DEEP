#!/bin/bash

CC=g++
CFLAGS= -ansi -O5 -Wall
LDFLAGS= -ansi -lm -Wall
EXEC=train predict
OBJ1=train.o predict.o deep.o

all: $(EXEC)

train : $(OBJ1) train.o
	$(CC) -o $@ $^ $(LDFLAGS)
predict : $(OBJ1) predict.o
	$(CC) -o $@ $^ $(LDFLAGS)
 
##########################################
# Generic rules
##########################################

%.o: %.cpp %.h
	$(CC) -o $@ -c $< $(CFLAGS)
	
%.o: %.cpp
	$(CC) -o $@ -c $< $(CFLAGS)
	
clean:
	rm -f *.o *~ $(EXEC)
