#pragma once

#ifdef USE_SCOREP
#include <scorep/SCOREP_User.h>
#else
#define SCOREP_USER_REGION_DEFINE(name)
#define SCOREP_USER_REGION_BEGIN(name, desc, type)
#define SCOREP_USER_REGION_END(name)
#endif
