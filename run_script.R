library(rmarkdown)
render("online_news_pop.Rmd", 
       output_file = "data_channel_is_entertainment.md",
       params = list(channel = "data_channel_is_entertainment"))

apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "online_news_pop.Rmd", output_file = x[[1]], params = x[[2]])
      }
)