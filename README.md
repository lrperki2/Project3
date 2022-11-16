# Purpose

The purpose of this repo is to demonstrate various predictive modeling methods 
on the [Online News Popularity](https://archive.ics.uci.edu/ml/datasets/online+news+popularity) data set from UCI and show how report automation can be accomplished via R 
Markdown.

# Requirements

The following packages were used to create the reports: 

  + `knitr`: used to document code in R Markdown format  
  + `tidyverse`: used for pipe operators, data manipulation, and plotting  
  + `caret`: used to train predictive models
  + `corrplot`: used to generate correlation plots
  + `randomForest`: used for random forest modeling
  + `gbm`: used for gradient boosted modeling

# Reports

  + The analysis for [Business articles is available here](data_channel_is_bus.html)  
  + The analysis for [Entertainment articles is available here](data_channel_is_entertainment.html)  
  + The analysis for [Lifestyle articles is available here](data_channel_is_lifestyle.html)  
  + The analysis for [Social Media articles is available here](data_channel_is_socmed.html)  
  + The analysis for [Technology articles is available here](data_channel_is_tech.html)
  + The analysis for [World news articles is available here](data_channel_is_world.html)

# R Markdown Automation

The reports are automated across a parameterized variable by running the 
`run_script.R` file containing the code below. First, the `rmarkdown` package 
is read in to be able to use the `render()` function, which knits the documents. 
The first argument is the file name of the R Markdown document to be rendered, 
the `output_file` argument takes the file path of a single parameterized report, 
and the `params` argument takes a list consisting of the single parameter name 
and value being referenced for the output file. Any acceptable value of the 
parameter can be used to generate a report for the specific level of that 
parameter, but this code must be run before the mass exporting code. This is 
because the `reports` object used to run all reports in series, contains a list 
of all the output files and parameter names, and is generated from the data 
values directly, and so is only created once the data has been read in, which is 
done by executing the code in the .Rmd file once.

The second block of code runs the reports sequentially for all levels of the 
parameter. The `apply()` function applies the `render()` function to every row 
of the `reports` object, which contains the output file name and parameter value 
of each level of the parameter. The `output_file` argument takes in the 
filename from `reports`, and `params` argument takes in the one element list of 
the parameter name value pair.

```
library(rmarkdown)
render("online_news_pop.Rmd", 
       output_file = "data_channel_is_entertainment.md",
       params = list(channel = "data_channel_is_entertainment"))

apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "online_news_pop.Rmd", output_file = x[[1]], params = x[[2]])
      }
)
```
