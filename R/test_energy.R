# demo for testing out functionality of energy statistics

# load package
require(energy)
require(ggplot2)
set.seed(52)


runsim <- function(x, y, title){

  dist <- dcov(x, y)
  corr <- dcor(x, y)
  
  df <- data.frame(x = x, y = y)
  
  
  titleStr <- title
  subTitleStr<- paste0("Distance Covariance: ", dist, " | Distance Correlation: ", corr)
  gg <- ggplot(df) + geom_point(aes(x=x, y=y)) + ggtitle(titleStr, subtitle = subTitleStr)
  
  
  plot(gg)
  
}

printsim <- function(x,y){
  dist <- dcov(x, y)
  corr <- dcor(x, y)
  
  print("--------- RESULTS -----------")
  print(paste0("Distance Covariance: ", dist))
  print(paste0("Distance Correlation: ", corr))
  print("-----------------------------")
  print("")
}
# step 1: simple distributions of a single variable (numeric values)


# part a: identical normal distributions
x <- rnorm(1000)
y <- rnorm(1000)

runsim(x, y, "Two identical normal distributions")
# part b: one normal, one uniform
x <- runif(1000)
y <- rnorm(1000)

runsim(x, y, "One normal, one uniform distribution")

# chi square + uniform
x <- rchisq(1000, 5)
y <- runif(1000)

runsim(x,y, "Normal and Chi Square (df = 5)")



# step 2: simple distributions of multiple variables (numeric values)

df1 <- data.frame(x=runif(100), y=rnorm(100))
df2 <- data.frame(x=rnorm(100), y-runif(100))

printsim(df1, df2)

# step 3