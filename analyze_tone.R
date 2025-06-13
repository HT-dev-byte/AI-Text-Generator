# analyze_tone.R
library(syuzhet)

args <- commandArgs(trailingOnly = TRUE)
text <- args[1]

words <- get_tokens(text)
sentiments <- get_sentiment(words, method = "syuzhet")

labels <- ifelse(sentiments > 0, "positive",
                 ifelse(sentiments < 0, "negative", "neutral"))
counts <- table(factor(labels, levels = c("positive", "negative", "neutral")))

cat(sprintf("positive:%d,negative:%d,neutral:%d\n", counts["positive"], counts["negative"], counts["neutral"]))
