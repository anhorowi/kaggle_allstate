#Exploratory

#######################################################
################ best linear model ####################
######################################################
library(ggplot2)
df <- read.csv('C:/Users/Will/Documents/MSA/fall3/machine_learning/kaggle_allstate/data/train.csv')

str(df)

n_levels <- sapply(df, function(x) length(levels(x)))

binary_var <- df[,n_levels == 2] #exract binary variables
binary_var <- data.frame(sapply(binary_var, function(x) as.numeric(x)-1))


# The ~80 binary variables are reduced to 40 principal components
pca <- prcomp(binary_var, scale=F) 
pca_summ <- summary(pca)$importance
plot(pca_summ[3,])
abline(v = 40)
new_vars <- data.frame(pca$x[,1:40]) # they are all converted to numeric

# all categorical variables ~ 700 to 200 using PCA
categorical <- model.matrix(~., data = df [2:117])
pca <- prcomp(categorical, scale = F)
pca_summ <- summary(pca)$importance
plot(pca_summ[3,])
abline(v = 200)
new_cat <- pca$x[,1:200]



#look for collinearity: The highest = 0.99,  Remove one of the variables
new_df <- cbind(data.frame(new_cat), df[,119:132])
df_cor <- NULL
for(i in 1:(ncol(new_df)-1)){
  for(j in (i+1):ncol(new_df)){
    name <- paste(names(new_df)[i], names(new_df)[j])
    correl <- cor(new_df[,i], new_df[,j])
    df_cor <- rbind(df_cor, data.frame(name, correl))
  }
}

df_cor <- df_cor[order(df_cor$correl, decreasing = T),]

new_df <- new_df[, -211]
linear_mod <- lm(loss ~., data = new_df)


# for(i in 1:length(new_df)){
# print(ggplot(new_df, aes(x = new_df[,i], y = loss)) + stat_binhex() + geom_smooth())
# readline()
# print(i)
# }

#look for quadratic terms
quadratic_df <- NULL
for(i in 1:(length(new_df)-1)){
  centered_quad <-new_df[,i]^2 - mean(new_df[,i]^2)
  mod <- lm(loss ~ new_df[,i] + centered_quad, data = new_df)
  p_val <- summary(mod)$coefficients[,4]
  p_val <- data.frame(t(as.matrix(p_val)))
  names(p_val) <- c('int', 'p_val_x', 'p_val_x^2')
  name <- data.frame(name = names(new_df)[i])
  add <- cbind(name, p_val)
  quadratic_df <- rbind(quadratic_df, add)
}

sig_quad <- quadratic_df[quadratic_df$`p_val_x^2` < .00000001,]
sig_quad <- sig_quad[complete.cases(sig_quad),]
sig_quad <- sig_quad[order(sig_quad$`p_val_x^2`),]
sig_quad <- sig_quad[1:100,]

#extract significant quadratic terms
quad_terms <- paste0('I(',sig_quad$name,'^2)', collapse = ' + ')


#look for significant interactions
interaction_df <- NULL
for(i in 1:(length(new_df)-2)){
  for(j in (i+1):(length(new_df)-1)){
    interaction = new_df[,i] * new_df[,j]
    mod <- lm(loss ~ new_df[,i] + new_df[,j] + interaction, data = new_df)
    p_val <- summary(mod)$coefficients[,4]
    p_val <- data.frame(t(as.matrix(p_val)))
    names(p_val) <- c('int', 'p_val_x', 'p_val_y', 'p_val_int')
    name_i <- data.frame(name = names(new_df)[i])
    name_j <- data.frame(name = names(new_df)[j])
    add <- cbind(name_i, name_j, p_val)
    interaction_df <- rbind(interaction_df, add)
  }
}

sig_int <- interaction_df
sig_int <- sig_int[order(sig_int$p_val_int),]
sig_int <- sig_int[1:100,]

int_terms <- paste0('I(', sig_int[,1],'*', sig_int[,2], ')', collapse = ' + ')
### Add in quadratic terms and interaction terms

all_terms <- paste('loss ~. ',quad_terms, int_terms, sep = ' + ')


#Linear model ain't so great
linear_mod <- lm(eval(parse(text = all_terms)), data = new_df)
#RSE 1984 on training


save.image()














