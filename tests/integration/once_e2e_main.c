#include <stdbool.h>
#include <stdint.h>

extern bool once_initially_incomplete(void);
extern bool once_execute_marks_complete(void);
extern int32_t once_execute_runs_once(void);

int main(void) {
  if (once_initially_incomplete())
    return 1;
  if (!once_execute_marks_complete())
    return 2;
  if (once_execute_runs_once() != 1)
    return 3;
  return 0;
}
