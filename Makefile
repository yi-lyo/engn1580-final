# Makefile: build the PortAudio program as `audio`
#
# Default build command requested:
#   gcc -o audio audio.c -lportaudio -lm
#
# Usage:
#   make            # builds ./audio
#   make run        # builds and runs ./audio
#   make clean      # removes build artifacts
#   make help       # shows targets/overrides

CC      ?= gcc
TARGET  := transmit
SRC     := transmit.c

# Keep flags minimal to match the provided compile command exactly.
# You can override/add flags from the command line, e.g.:
#   make CFLAGS="-O2 -Wall"
CFLAGS  ?= -O2 -Wall -Wextra
LDFLAGS ?=
LDLIBS  ?= -lportaudio -lm

.PHONY: all run clean help

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o

help:
	@printf "%s\n" \
	"Targets:" \
	"  all (default)  Build ./$(TARGET)" \
	"  run            Build then run ./$(TARGET)" \
	"  clean          Remove build artifacts" \
	"" \
	"Variables you can override:" \
	"  CC, CFLAGS, LDFLAGS, LDLIBS"
