library(rgl)
library(fda)
library(MASS)

binorm = function(x,y)
{
  z =  (15/ (2*pi)) * exp(-(1/2) * (x^2 + y^2))
  return(z)
}

# Set up data
samples = 10
x = seq(-3, 3, length.out = samples)
grid = cbind(rep(x, times=1, each=samples), rep(x, times=samples)) # cartesian product
Y = matrix(binorm(grid[,1], grid[,2]), nrow = samples, ncol = samples) # evaluate at each point

# Plot the surface
surface3d(x,x, Y, col = "green")

param.nbasis = 12
param.norder = 6

param.rangeval = c(min(x), max(x))
nbreaks = param.nbasis - param.norder + 2
basis.breaks = seq(param.rangeval[1], param.rangeval[2], length.out = nbreaks)

B.x = bsplineS(x, breaks = basis.breaks, norder = param.norder)
B.y = bsplineS(x, breaks = basis.breaks, norder = param.norder)

C = matrix(nrow = 10, ncol = param.nbasis^2)
for (i in 1:10) 
{
  C[i,] = kronecker(B.x[i,], B.y[i,])
}

B = ginv(t(C) %*% C) %*% t(C) %*% Y  #OLS
pred = C%*%B  #Predictions
surface3d(x,x, pred, col = "green")  #Plot 