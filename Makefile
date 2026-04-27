# Makefile — build transmit, receive, and test
#
# Usage:
#   make              # build ./transmit and ./receive
#   make transmit     # build only ./transmit
#   make receive      # build only ./receive
#   make test         # build ./test (no PortAudio or SDL2 required)
#   make run          # build and run ./transmit (reads from stdin)
#   make run-tests    # build and run ./test
#   make clean        # remove build artifacts
#   make help         # show this help

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra
LDFLAGS ?=

# test: pure signal-processing, no PortAudio or SDL2
LDLIBS_TEST ?= -lm

# transmit: PortAudio + math + optional SDL2/SDL2_ttf UI (-u)
LDLIBS_TRANSMIT ?= -lportaudio -lm $(SDL2_LIBS)

# receive: PortAudio + math + SDL2 + SDL2_ttf + pthreads
# pkg-config supplies the exact include paths and link flags for SDL2/TTF.
SDL2_CFLAGS  := $(shell pkg-config --cflags sdl2 SDL2_ttf 2>/dev/null)
SDL2_LIBS    := $(shell pkg-config --libs   sdl2 SDL2_ttf 2>/dev/null)
LDLIBS_RECEIVE ?= -lportaudio -lm -lpthread $(SDL2_LIBS)

TARGETS := transmit receive test

.PHONY: all run clean help

all: $(TARGETS)

transmit: transmit.c
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS_TRANSMIT)

receive: receive.c
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS_RECEIVE)

# test links only against libm — no PortAudio, no SDL2
test: test.c psk_common.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ test.c $(LDLIBS_TEST)

run: transmit
	./transmit

run-tests: test
	./test

clean:
	rm -f $(TARGETS) *.o
	rm -f /tmp/tx_test.bin /tmp/qpsk_test.bin /tmp/timing_test.bin

help:
	@printf "%s\n" \
	"Targets:" \
	"  all (default)  Build ./transmit, ./receive, and ./test" \
	"  transmit       Build only ./transmit" \
	"  receive        Build only ./receive" \
	"  test           Build ./test (no PortAudio / SDL2 needed)" \
	"  run            Build then run ./transmit (pipe data via stdin)" \
	"  run-tests      Build then run ./test" \
	"  clean          Remove build artifacts" \
	"" \
	"Variables you can override:" \
	"  CC, CFLAGS, LDFLAGS, LDLIBS_TRANSMIT, LDLIBS_RECEIVE, LDLIBS_TEST"
