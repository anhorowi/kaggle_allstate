
# Gradient boosting
#get the latest version
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
library(xgboost)

df <- read.csv('C:/Users/Will/Documents/MSA/fall3/machine_learning/kaggle_allstate/data/train.csv')
samp <- sample(1:nrow(df), round(.1*nrow(df)))
train <- df[-samp,]
response <- train[,length(train)]
train <- train[,-length(train)]
train <- model.matrix(~., data = train)

test <- df[samp,]

params <- list('objective = mae')

ln_cosh_obj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  grad <- tanh(preds-labels)
  hess <- 1-grad*grad
  return(list(grad = grad, hess = hess))
}

xg_mod <- xgboost(data = train, label = response,
                  nrounds = 2, objective = ln_cosh_obj)
x <- data(agaricus.train, package='xgboost')


resp <- predict(xg_mod, train)


#######################

library(Matrix)
library(xgboost)
library(Metrics)

train_path = "C:/Users/Will/Documents/MSA/fall3/machine_learning/kaggle_allstate/data/train.csv"
test_path = "C:/Users/Will/Documents/MSA/fall3/machine_learning/kaggle_allstate/data/test.csv"
submission_file_path = "C:/Users/Will/Documents/MSA/fall3/machine_learning/kaggle_allstate/data/sample_submission.csv"

train <- read.csv(train_path)
test <- read.csv(test_path)

y_train = log(train$loss)

train <- train[,-c(1,length(train))]
test <- test[,-1]

train_test <- rbind(train,test)
train_test <- model.matrix(~., data = train_test)
train <- train_test[1:nrow(train),]
test <- train_test[(nrow(train)+1):nrow(train_test),]
rm(train_test)

dtrain <- xgb.DMatrix(train, label=y_train)
dtest <- xgb.DMatrix(test)


xgb_params = list(
  seed = 0,
  colsample_bytree = 0.7,
  subsample = 0.7,
  eta = 0.075,
  objective = 'reg:linear',
  max_depth = 6,
  num_parallel_tree = 1,
  min_child_weight = 1,
  base_score = 7
)

xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}

res = xgb.cv(xgb_params,
             dtrain,
             nrounds=750,
             nfold=4,
             early_stopping_rounds=15,
             print_every_n = 10,
             verbose= 1,
             feval=xg_eval_mae,
             maximize=FALSE)

best_nrounds = res$best_iteration
cv_mean = res$evaluation_log$test_error_mean[best_nrounds]
cv_std = res$evaluation_log$test_error_std[best_nrounds]
cat(paste0('CV-Mean: ',cv_mean,' ', cv_std))

gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

submission = fread(SUBMISSION_FILE, colClasses = c("integer", "numeric"))
submission$loss = exp(predict(gbdt,dtest))
write.csv(submission,'xgb_starter_v2.sub.csv',row.names = FALSE)

#1126.19439 on PLB