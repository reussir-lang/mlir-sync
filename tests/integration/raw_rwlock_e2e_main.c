#include <stdbool.h>

extern bool rwlock_try_read_returns_true(void);
extern bool rwlock_try_write_returns_true(void);
extern bool rwlock_write_then_try_read_returns_false(void);
extern bool rwlock_read_then_try_write_returns_false(void);

int main(void) {
  if (!rwlock_try_read_returns_true())
    return 1;
  if (!rwlock_try_write_returns_true())
    return 2;
  if (rwlock_write_then_try_read_returns_false())
    return 3;
  if (rwlock_read_then_try_write_returns_false())
    return 4;
  return 0;
}
