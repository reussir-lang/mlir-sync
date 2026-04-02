#include <stdbool.h>

extern bool mutex_try_lock_returns_true(void);
extern bool mutex_relock_returns_false(void);
extern void mutex_lock_unlock_smoke(void);

int main(void) {
  if (!mutex_try_lock_returns_true())
    return 1;
  if (mutex_relock_returns_false())
    return 2;
  mutex_lock_unlock_smoke();
  return 0;
}
