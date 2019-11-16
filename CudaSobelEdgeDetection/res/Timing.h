#include "PreciseTimer.h"

#define MEASURE_TIME(reply, function_str, function)							        			\
		{																						\
			CPreciseTimer timer;																\
			for(int _timeri = 0; _timeri <= reply; _timeri++) {									\
				if (_timeri == 1)																\
					timer.StartTimer();															\
				function;																		\
			}																					\
			timer.StopTimer();																	\
			printf("Timer[%s]=%f\n", function_str, (float)timer.GetTimeMilliSec() / reply);     \
		}

void WarmUp();
