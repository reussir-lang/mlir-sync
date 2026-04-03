#include <stdint.h>

extern int64_t combining_lock_parallel_increment(void);

int main(void) {
  for (int attempt = 0; attempt < 32; ++attempt) {
    if (combining_lock_parallel_increment() != 4000)
      return attempt + 1;
  }
  return 0;
}
