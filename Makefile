# Makefile — build transmit and receive
#
# Usage:
#   make              # build both ./transmit and ./receive
#   make transmit     # build only ./transmit
#   make receive      # build only ./receive
#   make run          # build and run ./transmit (reads from stdin)
#   make clean        # remove build artifacts
#   make help         # show this help

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra
LDFLAGS ?=

# transmit: plain PortAudio + math
LDLIBS_TRANSMIT ?= -lportaudio -lm

# receive: PortAudio + math + SDL2 + SDL2_ttf + pthreads
# pkg-config supplies the exact include paths and link flags for SDL2/TTF.
SDL2_CFLAGS  := $(shell pkg-config --cflags sdl2 SDL2_ttf 2>/dev/null)
SDL2_LIBS    := $(shell pkg-config --libs   sdl2 SDL2_ttf 2>/dev/null)
LDLIBS_RECEIVE ?= -lportaudio -lm -lpthread $(SDL2_LIBS)

TARGETS := transmit receive

.PHONY: all run clean help

all: $(TARGETS)

transmit: transmit.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS_TRANSMIT)

receive: receive.c
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS_RECEIVE)

run: transmit
	./transmit

clean:
	rm -f $(TARGETS) *.o

help:
	@printf "%s\n" \
	"Targets:" \
	"  all (default)  Build ./transmit and ./receive" \
	"  transmit       Build only ./transmit" \
	"  receive        Build only ./receive" \
	"  run            Build then run ./transmit (pipe data via stdin)" \
	"  clean          Remove build artifacts" \
	"" \
	"Variables you can override:" \
	"  CC, CFLAGS, LDFLAGS, LDLIBS_TRANSMIT, LDLIBS_RECEIVE"
