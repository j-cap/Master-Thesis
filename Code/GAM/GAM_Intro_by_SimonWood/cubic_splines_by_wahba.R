

# read in the data
size<-c(1.42,1.58,1.78,1.99,1.99,1.99,2.13,2.13,2.13,
        2.32,2.32,2.32,2.32,2.32,2.43,2.43,2.78,2.98,2.98)
wear<-c(4.0,4.2,2.5,2.6,2.8,2.4,3.2,2.4,2.6,4.8,2.9,
        3.8,3.0,2.7,3.1,3.3,3.0,2.8,1.7)
x<-size-min(size);
x<-x/max(x)
plot(x,wear,xlab="Scaled engine size",ylab="Wear index")

# write a function defining the cubic regression splines recursively after Wahba
rk <- function(x,z) # R(x,z) for cubic spline on [0,1]
{
  ((z-0.5)^2-1/12)*((x-0.5)^2-1/12)/4 -
  ((abs(x-z)-0.5)^4-(abs(x-z)-0.5)^2/2 + 7/240)/24
}

spl.X<-function(x,xk)
# set up model matrix for cubic penalized regression spline
{ q<-length(xk)+2 # number of parameters
  n<-length(x) # number of data
  X<-matrix(1,n,q) # initialized model matrix
  X[,2]<-x # set second column to x
  X[,3:q]<-outer(x,xk,FUN=rk) # and remaining to R(x,xk)
  X
}

xk<-1:4/5 # choose some knots
X<-spl.X(x,xk) # generate model matrix
mod.1<-lm(wear~X-1) # fit model
xp<-0:100/100 # x values for prediction
Xp<-spl.X(xp,xk) # prediction matrix
lines(xp,Xp%*%coef(mod.1)) # plot fitted spline

