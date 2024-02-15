# the entire process was done in R Studio (Version 4.3.2)


# at first: restart R and clear Environment ----------------------------------------------------> 
# then select all and run
# after necessary packages are installed continue with rest of code step by step

# Packages needed for Topic modeling and sentiment analysis 
packages <- c("dplyr",
              "magrittr",
              "data.table",
              "tidytext",
              "topicmodels",
              "colorspace",
              "purrr",
              "ldatuning",
              "gmp",
              "wordcloud",
              "RColorBrewer",
              "ggplot2",
              "lubridate",
              "reshape2",
              "textmineR",
              "widyr", 
              "stringr", 
              "irlba",
              "textdata",
              "purrr")

# if need to install packages
for (i in 2:length(packages))
{
  install.packages(packages[i])
}
lapply(packages, require, character.only = TRUE)
# (it helped that some of the packages are loaded directly before using the part in which they are needed, especially when executing in multiple sessions)

# loading the data set - data acquisition
library(readxl)
data_pool <- read_excel(".../data_pool_products.xlsx")
View(data_pool)

# loading my selection of stop words
myStopwords <- read_excel(".../myStopwords.xlsx")
myStopwords2 <- myStopwords[,-1] #removing the column names out of Excel
remove(myStopwords)

# pre-processing the data --------------------------------------------------------------------------------------------------------------
library(stringr)
library(tools)
library(datasets)
library(methods)
# cleaning
data_pool$transcript <- gsub("\\b(\\w)[A-Z][^ ]*\\s*", " ", data_pool$transcript)  # removes all the CAPITALIZED WORDS like "SPEAKER"
data_pool$transcript <- gsub("[\r\n]", " ", data_pool$transcript)  # removes the line breaks
# tokenization
transcript_tokens <- data_pool %>% 
  tidytext::unnest_tokens(word, transcript)
transcript_tokens$word <- gsub('[[:digit:]]+', '', transcript_tokens$word) #removing digits
transcript_tokens$word <- gsub('[[:punct:]]+', '', transcript_tokens$word) #removing any form of punctuation or other arbitrary characters
# result so far: 22144 obs. of 4 variables 
# stop words removal
transcript_tokens <- transcript_tokens %>% filter(!(nchar(word) == 1)) %>% 
  anti_join(myStopwords2)
transcript_tokens <- transcript_tokens %>% filter(!(word==""))
# result after stop words removal: 13969 obs. of 4 variables 

# automated product modeling --------------------------------------------------------------------------------------------------------------
tokens <- transcript_tokens %>% mutate(ind = row_number()) # introduce a column with a number for each word
tokens <- tokens %>% group_by(id) %>% mutate(ind = row_number()) %>%
  tidyr::spread(key = ind, value = word) # group these numbers according to each video
# tokens$`word count` <- as.character(tokens$`word count`)
# tokens <- replace(tokens, is.na(tokens), "")
tokens <- tidyr::unite(tokens, text,-id,sep =" " ) # unite all individual words back to one line again as in raw data_pool
tokens$text <- trimws(tokens$text)

#create DTM (document term matrix), sparse matrix containing terms and documents as dimensions
dtm <- CreateDtm(tokens$text, 
                 doc_names = tokens$id, 
                 ngram_window = c(1, 2))
#explore the basic frequency
tf <- TermDocFreq(dtm = dtm)
original_tf <- tf %>% select(term, term_freq,doc_freq)
rownames(original_tf) <- 1:nrow(original_tf)
# Eliminate words appearing less than 2 times or in more than half of the
# documents
vocabulary <- tf$term[tf$term_freq > 1 & tf$doc_freq < nrow(dtm)/2]
dtm = dtm

# Running LDA 
k_list <- seq(1, 19, by = 1) # choose 19 as we have manually selected 19 different products (topics), including category "other"
model_dir <- paste0("models_", digest::digest(vocabulary, algo = "sha1"))
if (!dir.exists(model_dir)) dir.create(model_dir)
model_list <- TmParallelApply(X = k_list, FUN = function(k){
  filename = file.path(model_dir, paste0(k, "_topics.rda"))
  if (!file.exists(filename)) {
    m <- FitLdaModel(dtm = dtm, k = k, iterations = 500)
    m$k <- k
    m$coherence <- CalcProbCoherence(phi = m$phi, dtm = dtm, M = 19)
    save(m, file = filename)
  } else {
    load(filename)
  }
  m
}, export=c("dtm", "model_dir")) # export only needed for Windows machines

#model tuning
#choosing the best model
coherence_mat <- data.frame(k = sapply(model_list, function(x) nrow(x$phi)), 
                            coherence = sapply(model_list, function(x) mean(x$coherence)), 
                            stringsAsFactors = FALSE)
#plot coherence scores of 19 topics
ggplot(coherence_mat, aes(x = k, y = coherence)) +
  geom_point() +
  geom_line(group = 1)+
  ggtitle("Best Topic by Coherence Score") + theme_minimal() +
  scale_x_continuous(breaks = seq(1,19,1)) + xlab("number of topics") + ylab("Coherence")

#select models based on max average
model <- model_list[which.max(coherence_mat$coherence)][[ 1 ]]

#display top 20 terms based on phi value (not used)
model$top_terms <- GetTopTerms(phi = model$phi, M = 20) # here we want to display the top 20 words in each topic to compare them
top20_wide <- as.data.frame(model$top_terms)
# Define the text to be removed
text_to_remove <- c("na", "na_na")

# Apply filtering condition to each column
top20_wide <- top20_wide[apply(top20_wide, MARGIN = 1, function(row) !any(row == text_to_remove)), ]

#Visualizing of topics in a Dendrogram 
#probability distributions called Hellinger distance, distance between 2 probability vectors
model$topic_linguistic_dist <- CalcHellingerDist(model$phi)
model$hclust <- hclust(as.dist(model$topic_linguistic_dist), "ward.D")
model$hclust$labels <- paste(model$hclust$labels, model$labels[ , 1])
plot(model$hclust, labels = )

#2. word, topic relationship 
#looking at the terms allocated to the topic and their pr(word|topic)
allterms <- data.frame(t(model$phi))
allterms$word <- rownames(allterms)
rownames(allterms) <- 1:nrow(allterms)
allterms <- melt(allterms,idvars = "word") 
# allterms <- allterms %>% rename(topic = variable)
# FINAL_allterms <- allterms %>% group_by(topic) %>% arrange(desc(value))

#3. Topic,word,freq 
final_summary_words <- data.frame(top_terms = t(model$top_terms))
final_summary_words$topic <- rownames(final_summary_words)
rownames(final_summary_words) <- 1:nrow(final_summary_words)
final_summary_words <- final_summary_words %>% melt(id.vars = c("topic"))
final_summary_words <- final_summary_words %>% rename(word = value) %>% select(-variable)
final_summary_words <- left_join(final_summary_words,allterms)
final_summary_words <- final_summary_words %>% group_by(topic,word) %>%
  arrange(desc(value))
final_summary_words <- final_summary_words %>% group_by(topic, word) %>% filter(row_number() == 1) %>% 
  ungroup() %>% tidyr::separate(topic, into =c("t","topic")) %>% select(-t)
word_topic_freq <- left_join(final_summary_words, original_tf, by = c("word" = "term"))

#4. per-document-per-topic probabilities 
#trying to see the topic in each document
theta_df <- data.frame(model$theta)
theta_df$document <-rownames(theta_df) 
rownames(theta_df) <- 1:nrow(theta_df)
theta_df$document <- as.numeric(theta_df$document)
theta_df <- melt(theta_df,id.vars = "document")
theta_df <- theta_df %>% rename(topic = variable) 
theta_df <- theta_df %>% tidyr::separate(topic, into =c("t","topic")) %>% select(-t)
FINAL_document_topic <- theta_df %>% group_by(document) %>% 
  arrange(desc(value)) %>% filter(row_number() ==1)

#visualizing topics of words based on the max value of phi value in a pdf (each topic one page)
set.seed(1234) #selecting arbitrary seed as a coomon ground
pdf("Topic_Clusters.pdf")
for(i in 1:length(unique(final_summary_words$topic)))
{  wordcloud(words = subset(final_summary_words ,topic == i)$word, freq = subset(final_summary_words ,topic == i)$value, min.freq = 1,
             max.words=200, random.order=FALSE, rot.per=0.35, 
             colors=brewer.pal(8, "Dark2"))}
dev.off()


# Sentiment Analysis ----------------------------------------------------------------------------------------------------------------------------------

# loading necessary packages and dictionaries 
install.packages("textdata")
library(tidytext)
get_sentiments("afinn")
get_sentiments("bing")
get_sentiments("nrc")
library(janeaustenr)
library(dplyr)
library(stringr)
library(tidyr)
library(ggplot2)
library(wordcloud)
library(reshape2)

# creating a variable for positive sentiments out from nrc dictionary
nrc_positive <- get_sentiments("nrc") %>% 
  filter(sentiment == "positive")
View(nrc_positive)

# comparing the cleaned transcripts with positive sentiments (and only keeping those joints)
nrc_tokens <- transcript_tokens %>%
                inner_join(nrc_positive) %>%
                count(word, sort = TRUE) 
sum(nrc_tokens$n)

# displaying the net sentiment (positive - negative) for each bank
transcript_sentiment <- transcript_tokens %>%
  inner_join(get_sentiments("bing")) %>%
  count(bank, index = id %/% 1, sentiment) %>% # the smaller the index, the thinner the displayed columns 
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>% 
  mutate(sentiment = positive - negative)
ggplot(transcript_sentiment, aes(index, sentiment, fill = bank)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~bank, ncol = 5, scales = "free_x")

# Comparing the three sentiment dictionaries
afinn <- transcript_tokens %>% 
  inner_join(get_sentiments("afinn")) %>% 
  group_by(index = id %/% 10) %>% 
  summarise(sentiment = sum(value)) %>% 
  mutate(method = "AFINN")

bing_and_nrc <- bind_rows(
  transcript_tokens %>% 
    inner_join(get_sentiments("bing")) %>%
    mutate(method = "Bing et al."),
  transcript_tokens %>% 
    inner_join(get_sentiments("nrc") %>% 
                 filter(sentiment %in% c("positive", 
                                         "negative")), multiple = "all"
    ) %>%
    mutate(method = "NRC")) %>%
  count(method, index = id %/% 10, sentiment) %>%
  pivot_wider(names_from = sentiment,
              values_from = n,
              values_fill = 0) %>% 
  mutate(sentiment = positive - negative)

bind_rows(afinn, 
          bing_and_nrc) %>%
  ggplot(aes(index, sentiment, fill = method)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~method, ncol = 1, scales = "free_y")

# see how many positive and negative words are in these lexicons
get_sentiments("nrc") %>% 
  filter(sentiment %in% c("positive", "negative")) %>% 
  count(sentiment)
get_sentiments("bing") %>% 
  count(sentiment)

# Display most common positive and negative words
bing_general_counts <- transcript_tokens %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()
bing_general_counts
bing_general_counts %>%
  group_by(sentiment) %>%
  slice_max(n, n = 10) %>% 
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(n, word, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(x = "Contribution to sentiment",
       y = NULL)

# Display the most contributing words in a word cloud
transcript_tokens %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("gray20", "gray80"),
                   max.words = 50)




# Word Embedding Analysis ----------------------------------------------------------------------------------------------------------------------------------

library(widyr)
library(irlba)
library(broom)
library(plyr)
library(dplyr)

#create context window with length 5
tidy_skipgrams <- data_pool %>%
  unnest_tokens(ngram, transcript, token = "ngrams", n = 5) %>%
  mutate(ngramID = row_number()) %>% 
  tidyr::unite(skipgramID, id, ngramID) %>%
  unnest_tokens(word, ngram)

#calculate unigram probabilities (used to normalize skipgram probabilities later)
unigram_probs <- data_pool %>%
  unnest_tokens(word, transcript) %>%
  count(word, sort = TRUE) %>%
  mutate(p = n / sum(n))

#calculate probabilities
skipgram_probs <- tidy_skipgrams %>%
  pairwise_count(word, skipgramID, diag = TRUE, sort = TRUE) %>%
  mutate(p = n / sum(n))

#normalize probabilities
normalized_prob <- skipgram_probs %>%
  filter(n > 20) %>%
  rename(word1 = item1, word2 = item2) %>%
  left_join(unigram_probs %>%
              select(word1 = word, p1 = p),
            by = "word1") %>%
  left_join(unigram_probs %>%
              select(word2 = word, p2 = p),
            by = "word2") %>%
  mutate(p_together = p / p1 / p2)
head(normalized_prob)

#more useful output can be created by filtering this data frame for an individual word - let’s try “chase”:
normalized_prob %>% 
  filter(word1 == "chase") %>%
  arrange(-p_together) #as you can see we now can filter out the individual slogans

#plot all of the words in our model in multidimensional space
pmi_matrix <- normalized_prob %>%
  mutate(pmi = log10(p_together)) %>%
  cast_sparse(word1, word2, pmi)

search_synonyms <- function(word_vectors, selected_vector) {
  similarities <- word_vectors %*% selected_vector %>%
    tidy() %>%
    as_tibble() %>%
    rename(token = .rownames,
           similarity = unrowname.x.)
  similarities %>%
    arrange(-similarity)
}
pres_synonym <- search_synonyms(word_vectors,word_vectors["yeah",])
pres_synonym 

pmi_svd <- irlba(pmi_matrix, 2, maxit = 500)

#output the word vectors
word_vectors <- pmi_svd$u
rownames(word_vectors) <- rownames(pmi_matrix)

#grab 100 arbitrary words
forplot<-as.data.frame(word_vectors[450:500,])
forplot$word<-rownames(forplot)

#now plot
ggplot(forplot, aes(x=V1, y=V2, label=word))+
  geom_text(aes(label=word),hjust=0, vjust=0, color="blue")+
  theme_minimal()+
  xlab("First Dimension")+
  ylab("Second Dimension")


# Install and load the Rtsne package for t-SNE visualization
install.packages("Rtsne")
library(Rtsne)

# Assuming word_vectors is your high-dimensional word vectors matrix
# Set a seed for reproducibility
set.seed(42)

# Run t-SNE to reduce dimensionality to 2D
tsne_result <- Rtsne(word_vectors)

# Create a data frame with the results
tsne_df <- data.frame(word = rownames(word_vectors), 
                      X = tsne_result$Y[, 1], 
                      Y = tsne_result$Y[, 2])

# Now you can plot the words in the 2D space
plot(tsne_df$X, tsne_df$Y, pch = 19, cex = 0.7, 
     main = "t-SNE Visualization", 
     xlab = "Dimension 1", 
     ylab = "Dimension 2")

# Add labels for some words
text(tsne_df$X, tsne_df$Y, 
     labels = tsne_df$word, 
     pos = 1, cex = 0.6)

# Assuming tsne_result is your t-SNE result with two dimensions
X <- tsne_df[, 2]
Y <- tsne_df[, 3]
words <- tsne_df$word

# Create a subset of points within the specified region
subset_points <- (Y > -2) & (Y < 5) & (X > 7) & (X < 9)

# Plot the subset of points
plot(X[subset_points], Y[subset_points], type = "n", xlab = "Dimension 1", ylab = "Dimension 2")

# Add text labels for each word
text(X[subset_points], Y[subset_points], labels = words[subset_points], cex = 0.8)



# Construal Level Analysis ----------------------------------------------------------------------------------------------------------------------------------

# calculating how many words we have in total (all ids and all banks)
general_count <- transcript_tokens %>%
  count(word, id, bank, product)

# calculating the number of words in each video transcript
word_count <- data.frame()
for (count2 in 1:300) {
  output2 = c(sum(general_count$n[general_count$id == count2]))
  word_count = rbind(word_count, output2)
}
colnames(word_count) <- c("count")

# calculating the number of words and videos in each bank domain
bank_count <- data.frame()
for (count in 1:10) {
  output = c(sum(general_count$n[general_count$bank == count]))
  bank_count = rbind(bank_count, output)
}
colnames(bank_count) <- c("count")
bank_count$videos <- table(data_pool$bank)

# calculating the number of words and videos in each product domain
products <- c("bank (savings) account", 
              "mobile app/online banking",
              "(cash-back) credit (Master-) card",
              "business card",
              "digital banking assistant",
              "loan", 
              "trading software", 
              "investment broker",
              "bank job",
              "financial advisor",
              "car purchase software","other")
product_count <- data.frame()
for (count5 in products) {
  output5 = c(sum(general_count$n[general_count$product == count5]))
  product_count = rbind(product_count, output5)
}
colnames(product_count) <- c("count")
product_count$videos <- table(data_pool$product)

# importing concreteness dictionary 
concDic <- read.csv("Downloads/crr_concreteness.csv", 
                    col.names = c("word", "concreteness", "familiarity"))

# comparing the Concreteness Dictionary with my transcript_tokens - for products
construal_list_product <- inner_join(x = transcript_tokens, y = concDic, by = "word") %>%
  count(word, concreteness, familiarity, product)

# comparing the Concreteness Dictionary with my transcript_tokens - for banks
construal_list_bank <- inner_join(x = transcript_tokens, y = concDic, by = "word") %>%
  count(word, concreteness, familiarity, bank)

# comparing the Concreteness Dictionary with my transcript_tokens - for banks
construal_list_id <- inner_join(x = transcript_tokens, y = concDic, by = "word") %>%
  count(word, concreteness, familiarity, id)

# calculating the number of concreteness terms in each domain
number_conc_words_id <- data.frame()
for (count4 in 1:300) {
  output4 <- c(sum(construal_list_id$n[construal_list_id$id == count4]))
  number_conc_words_id = rbind(number_conc_words_id, output4)
}
colnames(number_conc_words_id) <- c("concreteness count")

number_conc_words_bank <- data.frame()
for (count3 in 1:10) {
  output3 <- c(sum(construal_list_bank$n[construal_list_bank$bank == count3]))
  number_conc_words_bank = rbind(number_conc_words_bank, output3)
}
colnames(number_conc_words_bank) <- c("concreteness count")

number_conc_words_product <- data.frame()
for (y in products) {
  output7 <- c(sum(construal_list_product$n[construal_list_product$product == y]))
  number_conc_words_product = rbind(number_conc_words_product, output7)
}
colnames(number_conc_words_product) <- c("concreteness count")

# calculating the three ratios 
constr_ratios_id <- data.frame(as.numeric(
  number_conc_words_id$`concreteness count` / word_count$count))
colnames(constr_ratios_id) <- c("transcript ratio")

constr_ratios_bank <- data.frame(as.numeric(
  number_conc_words_bank$`concreteness count` / bank_count$count))
colnames(constr_ratios_bank) <- c("bank domain ratio")
constr_ratios_bank$bank <- 1:10

constr_ratios_products <- data.frame(as.numeric(
  number_conc_words_product$`concreteness count` / product_count$count))
colnames(constr_ratios_products) <- c("product ratio")
constr_ratios_products$products <- products

# calculating the sum of the concreteness scores per video
construal_list_id$concr_total <- construal_list_id$concreteness * construal_list_id$n
concreteness_scores_video <- data.frame()
for (i in 1:300) {
  output6 <- as.numeric(sum(construal_list_id$concr_total[construal_list_id$id == i]))
  concreteness_scores_video = rbind(concreteness_scores_video, output6)
}
colnames(concreteness_scores_video) <- c("concreteness score per video")

# calculating the average by dividing by the count of concreteness words in each video 
concreteness_scores_video$average <- 
  concreteness_scores_video$`concreteness score per video` / number_conc_words_id #these values are now of type 'list'
concreteness_scores_video$average <- as.numeric(
  unlist(concreteness_scores_video$average)) #convert type list to numeric vector
concreteness_scores_video$id <- 1:300 #creating a new column id in order to perform hypothesis test on id's
concreteness_scores_video$id <- as.factor(concreteness_scores_video$id) #convert 'id' to type factor for the post hoc test

# now, in order to compare these average values with each other, I multiply each with the 'transcript ratio'
concreteness_scores_video$`comparison value` <- as.numeric(
  concreteness_scores_video$average * constr_ratios_id$`transcript ratio`)

# Creating a line plot with all absolute video scores
ggplot(concreteness_scores_video, 
       aes(x = 1:nrow(concreteness_scores_video), 
           y = `concreteness score per video`)) +
  geom_point() +
  labs(title = "Concreteness Scores - absolute",
       x = "Video number",
       y = "Absolute values") +
  scale_color_manual(values = c("black"))


# Creating a line plot with the average of the scores
ggplot(concreteness_scores_video, 
       aes(x = 1:nrow(concreteness_scores_video), 
           y = average)) +
  geom_point() +
  geom_smooth(method = NULL, se = FALSE, color = "red") +
  labs(title = "Concreteness Scores - relative",
       x = "Video number",
       y = "Average values")

# Creating a box plot with the comparison values
ggplot(concreteness_scores_video, aes(x = 1, y = `comparison value`)) +
  geom_boxplot(fill = "white") +
  labs(title = "Box Plot of Comparable Concreteness Scores", x = "", y = "Comparison Value")

summary(concreteness_scores_video)
sd(concreteness_scores_video$average)


# Shapiro test: testing for normal distribution of the average scores
print(shapiro.test(concreteness_scores_video$average)$p.value)
# for visual inspection
hist(concreteness_scores_video$average)

# Display the p-values
print(normality_results)

# Perform Wilcoxon signed-rank test
print(wilcox.test(concreteness_scores_video$average - 0.5, mu = 0, alternative = "greater"))

# Hypothesis testing - one sample t-test
threshold <- 50  # Set your threshold value
# One-sample t-test
print(t.test(concreteness_scores_video$average, mu = threshold))

# comparing the Concreteness Scores for banks ----------------------------------

# add column 'bank' and directly convert to type factor
concreteness_scores_video$bank <- as.factor(data_pool$bank)

# calculate the product of Bank Domain Ratio and corresponding values of concreteness scores average
bank_comparison <- data.frame()
for (j in 1:10) {
  result <- sum(concreteness_scores_video$`comparison value`[concreteness_scores_video$bank == j])
  bank_comparison = rbind(bank_comparison, result)
}
colnames(bank_comparison) <- c("sums")
bank_comparison$`per video` <- bank_comparison$sums / bank_count$videos
#bank_comparison$comparison <- bank_comparison$`per video` * constr_ratios_bank$`bank domain ratio`
bank_comparison$bank <- as.factor(1:10)

# Hypothesis testing - ANOVA to compare construal scores among banks
anova_result_bank <- aov(`per video` ~ bank, data = bank_comparison)
summary(anova_result_bank)
# output: summary(anova_result_bank)
# Df    Sum Sq    Mean Sq
# bank   9  71.68   7.965

# Post hoc tests (e.g., Tukey's HSD) if ANOVA is significant
posthoc <- TukeyHSD(anova_result_bank)
print(posthoc)
#Result: NaNs produced --> no significant evidence to reject the null hypothesis

sd(bank_comparison$`per video`)
mean(bank_comparison$`per video`)

# Importing the names of the banks for better visualization
bank_comparison$names <- c("JP Morgan Chase", "Bank of America", "Citigroup",
                           "Wells Fargo", "Goldman Sachs", "Morgan Stanley",
                           "U.S. Bancorp", "TD Bank", "PNC Financial Services", 
                           "Capital One")

# Box plot to visualize the absolute distribution of construal scores among banks
ggplot(bank_comparison, aes(x = names, y = sums)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 60, hjust = 1.1, vjust = 1)) +
  labs(title = "Construal Level Analysis Among Banks",
       x = "Banks",
       y = "Concreteness Score per Bank")

# Create a scatter plot with a regression line
ggplot(bank_comparison, 
       aes(x = 1:nrow(bank_comparison), 
           y = as.numeric(`per video`))) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Construal Level Analysis Among Banks - relative",
       x = "Banks",
       y = "Relative Concreteness Scores per Bank ")


# comparing the Concreteness Scores for products -------------------------------
concreteness_scores_video$product <- data_pool$product
  
product_comparison <- data.frame()
for (z in products) {
  result2 <- sum(concreteness_scores_video$average[concreteness_scores_video$product == z])
  product_comparison = rbind(product_comparison, result2)
}
colnames(product_comparison) <- c("sums")
product_comparison$comparison <- product_comparison$sums / product_count$videos
product_comparison$product <- sort(products)

# Hypothesis testing - ANOVA to compare construal scores among products
anova_result_products <- aov(comparison ~ product, data = product_comparison)
summary(anova_result_products)
# output: summary(anova_result_products)
#             Df Sum Sq Mean Sq
# product     11 224469   20406

# Post hoc tests for products
posthoc_products <- TukeyHSD(anova_result_products)
print(posthoc_products)
#Result: NaNs produced --> no significant evidence to reject the null hypothesis

sd(product_comparison$comparison)
mean(product_comparison$comparison)

# Box plot to visualize the distribution of construal scores among products
ggplot(product_comparison, aes(x = product, y = sums)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 60, hjust = 1.1, vjust = 1)) +
  labs(title = "Construal Level Analysis among Products",
       x = "Products",
       y = "Concreteness Score per Product")

# Create a scatter plot with a regression line
ggplot(product_comparison, 
       aes(x = 1:nrow(product_comparison), 
           y = as.numeric(comparison))) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Construal Level Analysis Among Products - relative",
       x = "Products",
       y = "Relative Concreteness Scores per Product ")


# Gender Analysis ----------------------------------------------------------------------------------------------------------------------------------

install.packages("rjson")
library(rjson)
library(jsonlite)
library(dplyr)
library(tidyr)

# importing a usable gender words dictionary
gDic <- fromJSON("Downloads/gendered_words.json")
gDic$word <- gsub("_"," ", gDic$word, perl = FALSE) #removing the underscore of some observations

# # comparing the cleaned transcripts with Gender Dictionary (and only keeping those joints)
# gender_tokens <- transcript_tokens %>%
#   inner_join(gDic, relationship = "many-to-many") %>%
#   count(word, gender, sort = TRUE)
# similarly comparing the words but by each video
gender_tokens_id <- transcript_tokens %>%
  inner_join(gDic, relationship = "many-to-many") %>%
  count(gender, id, sort = TRUE)

# calculating the percentage of total masculine and feminine words in each video ad
gender_count <- data.frame()
for (i in 1:300) {
  result3 = c(sum(gender_tokens_id$n[gender_tokens_id$id == i]))
  gender_count = rbind(gender_count, result3)
}
colnames(gender_count) <- c("count")
gender_count <- gender_tokens_id %>%
  spread(key = gender, value = n, fill = 0) %>%
  select(id, f, m, n)
gender_count[is.na(gender_count)] <- 0 #replace NAs with 0 in the resulting data frame

word_count$id <- 1:300

# Merge gender_count and word_count data frames
merged_count <- merge(gender_count, word_count, by = "id")

# Calculate the percentage of masculine and feminine words
merged_count$masculine_percentage <- (merged_count$m / merged_count$count) * 100
merged_count$feminine_percentage <- (merged_count$f / merged_count$count) * 100
result_data <- merged_count[, c("id", "masculine_percentage", "feminine_percentage")]
# Remove rows where all values are 0
result_data <- subset(result_data, !(masculine_percentage == 0 & feminine_percentage == 0))

# Check the correlation between masculine and feminine scores
print(cor.test(result_data$masculine_percentage, result_data$feminine_percentage))
#	results: Pearson's product-moment correlation
# data:  result_data$masculine_percentage and result_data$feminine_percentage
# t = -3.3334, df = 54, p-value = 0.001555
# alternative hypothesis: true correlation is not equal to 0
# 95 percent confidence interval:
#   -0.6097792 -0.1685037
# sample estimates:
#   cor 
# -0.4131038 

# In order for the t-test to be meaningful, I test each column for normal distribution
print(sapply(result_data, function(x) shapiro.test(x)$p.value))
#results:  id     masculine_percentage  feminine_percentage 
#.   1.456359e-0    8.407552e-11         2.648168e-08 
# this means that the data actually is not normally distributed, hence t-test does not work here
# Hence, I perform the Wilcoxon signed-rank test
print(wilcox.test(result_data$masculine_percentage, result_data$feminine_percentage, paired = FALSE))
# the results of the WRS test revealed a significant difference
# now generic statistics are used to inspect in which way the data leans 
summary(result_data)
# masculine mean: 3.657; feminine mean: 1.607  

# Conduct a new test for the additional hypothesis
# Create a chi-square contingency table
contingency_table <- table(result_data$masculine_percentage > 0, result_data$feminine_percentage > 0)

# Perform chi-square test
chi_square_result <- chisq.test(contingency_table)

# Print the results
print(chi_square_result)

# comparing the values for banks -----------------------------------------------

# Merge 'result_data' with 'data_pool' based on the 'id' column
merged_result <- merge(result_data, data_pool[, c('id', 'bank')], by = 'id', all.x = TRUE)
# The resulting data frame 'merged_result' will have an additional 'bank' column

# Calculate the count of videos for each bank
bank_counts = cbind(table(merged_result$bank)/bank_count$videos)
print(bank_counts)
