#include <stdint.h>

extern int64_t typed_rwlock_parallel_increment(void);

int main(void) {
  for (int attempt = 0; attempt < 32; ++attempt) {
    if (typed_rwlock_parallel_increment() != 4000)
      return attempt + 1;
  }
  return 0;
}
