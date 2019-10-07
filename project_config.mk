# Sources
SOURCE_DIRS := src lib

# Object directory
OBJECT_DIR := obj

# Resulting binary
OUTPUT_FILE := bin/wadj
LINKER_MODE := CXX

# Dependency set name
LIBRARY_PACK_NAME := imp-re_deps_v1.0
USED_PACKAGES := openal freetype2 ogg vorbis vorbisenc vorbisfile zlib fmt
USED_EXTERNAL_PACKAGES :=
ifeq ($(TARGET_OS),windows)
USED_PACKAGES += sdl2
else
USED_EXTERNAL_PACKAGES += sdl2
endif


# Flags
CXXFLAGS := -Wall -Wextra -pedantic-errors -std=c++2a
LDFLAGS :=
# Important flags
override CXXFLAGS += -include src/utils/common.h -include src/program/parachute.h -Isrc -Ilib/include $(subst -Dmain,-DENTRY_POINT,$(sort $(deps_compiler_flags)))
override CXXFLAGS += -Ilib/include/cglfl_gl3.2_core # OpenGL version
override LDFLAGS += $(filter-out -mwindows,$(deps_linker_flags))

# Build modes
$(call new_mode,debug)
$(mode_flags) CXXFLAGS += -g -D_GLIBCXX_ASSERTIONS

$(call new_mode,debug_hard)
$(mode_flags) CXXFLAGS += -g -D_GLIBCXX_DEBUG

$(call new_mode,release)
$(mode_flags) CXXFLAGS += -DNDEBUG -O3
$(mode_flags) LDFLAGS += -O3 -s
ifeq ($(TARGET_OS),windows)
$(mode_flags) LDFLAGS += -mwindows
endif

# File-specific flags
FILE_SPECIFIC_FLAGS := lib/implementation.cpp lib/cglfl.cpp > -g0 -O3

# Precompiled headers
PRECOMPILED_HEADERS := src/game/*.cpp > src/game/master.hpp

# Custom targets
.PHONY: package
package:
	rm -rf ./LD45_WADJ/
	cp -af ./bin ./LD45_WADJ/
	find ./LD45_WADJ/ -type f -name "*.mmp" -print -delete
	find ./LD45_WADJ/ -type f -name "_*" -print -delete
	rm -rf ./LD45_WADJ/assets/_images
	tree ./LD45_WADJ/
	rm -f LD45_WADJ.zip
	zip -q -r LD45_WADJ.zip LD45_WADJ/
	rm -rf ./LD45_WADJ/

# Code generation
GEN_CXXFLAGS := -std=c++2a -Wall -Wextra -pedantic-errors
override generators_dir := gen
override generated_headers := src/utils/mat.h src/utils/macro.h
override generate_file = $(call host_native_path,$1) : $(generators_dir)/make_$(subst .,_,$(notdir $1)).cpp ; \
	@+$(MAKE) -f gen/Makefile _gen_dir=$(generators_dir) _gen_target_file=$1 --no-print-directory
$(foreach f,$(generated_headers),$(eval $(call generate_file,$f)))
