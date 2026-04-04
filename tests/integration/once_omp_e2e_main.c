#include <stdint.h>

extern int32_t once_parallel_runs_once(void);

int main(void) {
  for (int attempt = 0; attempt < 32; ++attempt) {
    if (once_parallel_runs_once() != 1)
      return attempt + 1;
  }
  return 0;
}
