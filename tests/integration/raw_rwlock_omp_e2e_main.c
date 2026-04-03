#include <stdint.h>

extern int64_t raw_rwlock_parallel_read_sum(void);

int main(void) {
  for (int attempt = 0; attempt < 32; ++attempt) {
    if (raw_rwlock_parallel_read_sum() != 28000)
      return attempt + 1;
  }
  return 0;
}
