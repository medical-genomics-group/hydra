#ifndef SRC_BAYESW_ARMS_H_
#define SRC_BAYESW_ARMS_H_



int arms_simple (int ninit, double *xl, double *xr,
                 double (*myfunc)(double x, void *mydata), void *mydata,
                 int dometrop, double *xprev, double *xsamp);

int arms (double *xinit, int ninit, double *xl, double *xr,
          double (*myfunc)(double x, void *mydata), void *mydata,
          double *convex, int npoint, int dometrop, double *xprev, double *xsamp,
          int nsamp, double *qcent, double *xcent, int ncent,
          int *neval);

double expshift(double y, double y0);

#define YCEIL 50.                /* maximum y avoiding overflow in exp(y) */

#endif /* SRC_BAYESW_ARMS_H_ */
