df0 <- input1 %>% group_by(year) %>% summarise(mean = sum(TotalVolume))
