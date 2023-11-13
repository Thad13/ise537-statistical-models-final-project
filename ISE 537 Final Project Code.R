#### ISE 537 Final Project
#### Author: Thad M.


# Load Packages
library(tidyverse)
library(Lahman) ## This package contains the dataset
library(glmnet)
library(leaps)
library(Metrics)
library(olsrr)
library(broom)

## The data frame Teams comes directly from the Lahman package.
## It contains all of the data that we will use for this project. 

# Read Data Into R
set.seed(2)
Teams

# Filter the data to only look at data from recent years
teams <- Teams %>% 
  filter(yearID >= 2000) %>% 
  filter(yearID != 2020)

# Exploratory Data Analysis
plot(teams$AB, teams$W, xlab = "At Bats", ylab = "Wins", main = "Scatterplot of Wins vs At Bats without 2020")
plot(teams$H, teams$W, xlab = "Hits", ylab = "Wins", main = "Scatterplot of Wins vs Hits for all Years")
plot(teams$X2B, teams$W)
plot(teams$RA, teams$W, xlab = "Runs Allowed", ylab = "Wins", main = "Scatterplot of Wins vs Runs Allowed without 2020")
plot(teams$ER, teams$W)
plot(teams$attendance, teams$W)

# Split the data into training and testing data
test_rows <- sample(nrow(teams), 0.2 * nrow(teams))
teams_train <- teams[-test_rows,]
teams_test <- teams[test_rows,]

# Run Simple Linear Regression with All Variables
modelfull <- lm(W ~ AB + H + X2B + X3B + HR + BB + SO + SB + CS + HBP + SF + RA + ER + ERA + SV + HA + HRA + BBA + SOA + E + DP + attendance, data = teams_train)
summary(modelfull)

model1 <- lm(W ~ AB + H + X2B + X3B + HR + BB + SO + SB + CS + HBP + SF + RA + ER + ERA + SV + HA + HRA + BBA + SOA + E + DP + attendance, data = teams)
plot(fitted(model1), rstandard(model1), main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")

# Checking assumptions of linear regression
par(mfrow = c(1, 3))
qqnorm(rstandard(modelfull))
qqline(rstandard(modelfull), col = "red", lwd = 2)
hist(rstandard(modelfull), col = "lightblue", main = "Histogram of Standardized Residuals", xlab = "Standardized Residuals")
plot(fitted(modelfull), rstandard(modelfull), main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
abline(a = 0, b = 0, col = "red", lwd = 2)

# Stepwise Regression
intercept <- lm(W ~ 1, data = teams_train)
n <- nrow(teams_train)
step(intercept, scope = list(lower = intercept, upper = modelfull), direction = "forward", k = log(n))

model_forward <- lm(formula = W ~ SV + BB + RA + H + HR + AB + SF + HBP + SOA, data = teams_train)
summary(model_forward)

step(modelfull, scope = list(lower = intercept, upper = modelfull), direction = "backward", k = log(n))

model_backward <- lm(formula = W ~ AB + H + X2B + HR + BB + SB + CS + HBP + RA +  ER + ERA + SV + HA + BBA, data = teams_train)
summary(model_backward)

step(intercept, scope = list(lower = intercept, upper = modelfull), direction = "both", k = log(n))

model_both <- lm(formula = W ~ SV + BB + RA + H + HR + AB + SF + HBP + SOA,  data = teams_train) 
summary(model_both)

qqnorm(rstandard(model_forward))
qqline(rstandard(model_forward), col = "red", lwd = 2)
hist(rstandard(model_forward), col = "lightblue")
plot(fitted(model_forward), rstandard(model_forward))
abline(a = 0, b = 0, col = "red", lwd = 2)

# Full Search Method
response <- data.matrix(teams_train %>%
                          select(W))
predictors <- data.matrix(teams_train %>%
                            select(AB, H, X2B, X3B, HR, BB, SO, SB, CS, HBP, SF, RA, ER, ERA, SV, HA, HRA, BBA,  SOA, E, DP, attendance))

compare_full <- leaps(predictors, t(response), method = "Cp", nbest = 1)
cbind(as.matrix(compare_full$which), compare_full$Cp)

model_search <- lm(W ~ AB + H + X2B + HR + BB + SB + CS + HBP + SF + RA + ER + ERA + SV + HA + HRA + BBA + E, data = teams_train)
summary(model_search)

# LASSO
lasso_cv <- cv.glmnet(predictors, response, alpha = 1, nfolds = 10)

lasso <- glmnet(predictors, response, alpha = 1, nlambda = 100)
coef(lasso, s = lasso_cv$lambda.min)

model_lasso <- lm(W ~ AB + H + X2B + X3B + HR + BB + SO + SB + CS + HBP + SF + RA + ERA + SV + HA + HRA + BBA + SOA + E + DP + attendance, data = teams_train)
summary(model_lasso)

# Calculating RMSE
actual <- teams_test$W
test_predictors <- data.matrix(teams_test %>%
              select(AB, H, X2B, X3B, HR, BB, SO, SB, CS, HBP, SF, RA, ER, ERA, SV, HA, HRA, BBA,  SOA, E, DP, attendance))

predicted_full <- predict(modelfull, teams_test)
predicted_forward <- predict(model_forward, teams_test)
predicted_backward <- predict(model_backward, teams_test)
predicted_search <- predict(model_search, teams_test)
predicted_lasso <- predict(model_lasso, teams_test)

rmse_full <- rmse(actual, predicted_full)
rmse_forward <- rmse(actual, predicted_forward)
rmse_backward <- rmse(actual, predicted_backward)
rmse_search <- rmse(actual, predicted_search)
rmse_lasso <- rmse(actual, predicted_lasso)

# Calculating Mallow's Cp
ols_mallows_cp(modelfull, modelfull)
ols_mallows_cp(model_forward, modelfull)
ols_mallows_cp(model_backward, modelfull)
ols_mallows_cp(model_search, modelfull)
ols_mallows_cp(model_lasso, modelfull)

# Calculating AIC
extractAIC(modelfull, k = 2)[2]
extractAIC(model_forward, k = 2)[2]
extractAIC(model_backward, k = 2)[2]
extractAIC(model_search, k = 2)[2]
extractAIC(model_lasso, k = 2)[2]

# Calculating BIC
extractAIC(modelfull, k = log(n))[2]
extractAIC(model_forward, k = log(n))[2]
extractAIC(model_backward, k = log(n))[2]
extractAIC(model_search, k = log(n))[2]
extractAIC(model_lasso, k = log(n))[2]

# Additional Residual Analysis
qqnorm(rstandard(model_search))
qqline(rstandard(model_search), col = "red", lwd = 2)
hist(rstandard(model_search), col = "lightblue", main = "Histogram of Standardized Residuals", xlab = "Standardized Residuals")
plot(fitted(model_search), rstandard(modelfull), main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals")
abline(a = 0, b = 0, col = "red", lwd = 2)

