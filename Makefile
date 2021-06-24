CC=gcc
CFLAGS=-I./lib/estimator
CFLAGS+=-O3
CFLAGS+=-L/usr/local/lib/libestimator
CFLAGS+=-lestimator

SRC=main.c
EXE=pwm_estimator

ALL:
	${CC} ${SRC} -o ${EXE} ${CFLAGS}

.PHONY: clean
clean:
	rm ${EXE}
