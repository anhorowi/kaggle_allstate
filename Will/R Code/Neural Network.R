#Neural Network

library(ggplot2)
library(h2o)
library(h2o)

## Start a local cluster with 2GB RAM
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, 
                    max_mem_size = '2g')

df <- read.csv('C:/Users/Will/Documents/MSA/fall3/machine_learning/kaggle_allstate/data/train.csv')

str(df)
normalize <- function(x){
  return((x - min(x))/(max(x)-min(x)))
}
#create binary variables
categorical_df <- data.frame(model.matrix(~., data = df [2:117]))
normalized_df <- df[,118:(length(df)-1)]
normalized_df <- data.frame(sapply(normalized_df, normalize))
response <- df$loss
new_df <- cbind(categorical_df, normalized_df, response)
df_h2o <- as.h2o(new_df)


neural_net_model <-  h2o.deeplearning(x = names(df_h2o)[1:(length(new_df)-1)],  # column numbers for predictors
                   y = 'response',   # column number for label
                   training_frame = df_h2o, # data in H2O format
                   activation = "Tanh", # or 'Tanh'
                   #input_dropout_ratio = 0.2, # % of inputs dropout
                   #hidden_dropout_ratios = c(0.5,0.5,0.5), # % for nodes dropout
                   hidden = c(50,50,50), # three layers of 50 nodes
                   epochs = 100) # max. no. of epochs

#read vignette  http://h2o-release.s3.amazonaws.com/h2o/rel-slater/9/docs-website/h2o-docs/booklets/DeepLearning_Vignette.pdf