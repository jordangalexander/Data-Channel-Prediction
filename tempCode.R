# training data
train = sample(1:nrow(news), 24890)
trainingData = subset(news, is_weekend == 0)
testData = subset(news, is_weekend == 1)
usable = as.data.frame(sample(1:nrow(news), 500))

# linear model
mod <- lm(rate_positive_words ~ avg_positive_polarity + max_negative_polarity
          + data_channel_is_entertainment + average_token_length + n_non_stop_unique_tokens + n_non_stop_words
          + lda_00 + lda_01 +lda_02 + lda_03 + lda_04 + global_subjectivity, data = news)
summary(mod)

# explore best predictors 
require(gbm)
boost.news = gbm(data_channel_is_bus ~ .-url, data = news, distribution = "gaussian", n.trees = 10000, shrinkage = 0.01, interaction.depth = 4)
summary(boost.news)

# Logistic Regression Model
glm.fit <- glm(data_channel_is_bus ~ lda_00 + lda_03 + lda_04 + kw_max_avg + kw_avg_max + kw_avg_avg + timedelta + num_imgs + global_subjectivity, data = news, family = "binomial")
summary(glm.fit)

exp(glm.fit$coef) # odds ratios
exp(confint.default(glm.fit)) # confidence intervals

# predictive probabilities
glm.probs = predict(glm.fit, type = "response") 
glm.probs[1:5]
mean(glm.probs)
glm.pred = ifelse(glm.probs > .15, 1, 0)

table(glm.pred, news$data_channel_is_bus) # confusion matrix
mean(glm.pred == news$data_channel_is_bus) # percentage correctly predicted

# LDA
boost.news = gbm(num_keywords ~ .-url, data = news, distribution = "gaussian", n.trees = 10000, shrinkage = 0.01, interaction.depth = 4)
summary(boost.news)

lda.fit = lda(num_keywords ~ lda_01 + lda_00 + lda_02 + lda_03 + lda_04 + kw_min_max + kw_max_min + kw_max_avg + data_channel_is_socmed + kw_min_avg + data_channel_is_socmed + n_non_stop_words + kw_avg_min, data = trainingData)
lda.fit

lda.pred = predict(lda.fit, testData)
head(lda.pred)
table(lda.pred$class, testData$num_keywords) # confusion matrix
mean(lda.pred$class == testData$num_keywords) # average correcly predicted

# quadratic discriminant analysis
qda.fit = qda(num_keywords ~ kw_min_max + kw_avg_min + kw_min_max + kw_avg_min + kw_max_avg, data = trainingData)
qda.fit

qda.class = predict(qda.fit, testData)
table(qda.class$class, testData$num_keywords)
mean(qda.class$class == testData$num_keywords)

# KNN
attach(news)
Xlag = cbind(kw_min_max, kw_avg_min, kw_min_max, kw_avg_min, kw_max_avg)
train = news$is_weekend == 0
knn.pred = knn(Xlag[train,], Xlag[!train,], num_keywords[train], use.all = TRUE)
mean(knn.pred == num_keywords[!train])


# decision tree
require(tree)
attach(news)
tree.news = tree(as.factor(data_channel_is_bus) ~ .-url-data_channel_is_entertainment-data_channel_is_lifestyle-data_channel_is_socmed-data_channel_is_tech-data_channel_is_world, trainingData)
summary(tree.news)
plot(tree.news)
text(tree.news, pretty = 0)
plot(tree.news);text(tree.math, pretty = 0)
tree.pred = predict(tree.news, testData, type = "class")
with(testData, table(tree.pred, data_channel_is_bus)) # confusion matrix
4917/5190 # model accuracy

# prune the tree
cv.news = cv.tree(tree.news, FUN = prune.misclass)
plot(cv.news)
prune.news = prune.misclass(tree.news, best = 4)
plot(prune.news);text(prune.news, pretty = 0)

# predict on pruned tree
tree.pred = predict(prune.news, testData, type = "class")
with(testData, table(tree.pred, data_channel_is_bus))
4917/5190 # model accuracy
              