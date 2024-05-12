# [Monteprediction.com](https://monteprediction.com) Predictor #1 - Simple IV

Author: Rusty Conover (rusty@conover.me)

Date: 2024-05-12

## Overview

The script is designed to provide 1 million weekly return scenarios for 11 SPDR ETFs. These scenarios are generated based on the implied volatilities of each ETF, which are used to simulate returns.

## Method

The software utilizes [implied volatilities](https://www.investopedia.com/terms/i/iv.asp) for each ETF to produce simulated returns. It takes into account the available options data for each ETF, performing interpolation and filtering where necessary.

It is run by GitHub actions every Sunday morning.

## Caveats

1. The tool does not use a trading calendar, assuming returns are applied for five consecutive days.
2. Due to limited liquidity in some ETFs' options chains, the software performs interpolation and filtering of the available options data.
3. Implied volatility typically overstates actual volatility, as put options generally have [skew](https://www.optionseducation.org/news/volatility-skew-and-options-an-overview) factored into them. The tool does not include this skew in the calculation of the at-the-money implied volatility. Additionally, it should be noted that call options may still be overpriced, though the extent of this overpricing is not determined by the software.

## License

MIT License
