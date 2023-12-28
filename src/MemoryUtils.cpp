#include "MemoryUtils.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <unistd.h>
#include <sys/resource.h>
long getPeakMemoryUsage() {
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    return r_usage.ru_maxrss / 1024;
}

long getCurrentMemoryUsage() {
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
    return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE) / 1024 / 1024;
}