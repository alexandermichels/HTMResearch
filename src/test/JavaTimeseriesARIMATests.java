package test;

import com.github.signaflo.timeseries.forecast.Forecast;
import com.github.signaflo.timeseries.model.Model;
import com.github.signaflo.timeseries.model.arima.Arima;
import com.github.signaflo.timeseries.model.arima.ArimaOrder;
import static com.github.signaflo.data.visualization.Plots.plot;

import org.junit.Test;



public class JavaTimeseriesARIMATests 
{
	@Test
	public void SignaflowARIMAModelsWikiCode()
	{
		TimeSeries timeSeries = TestData.debitcards;
		Arima model = Arima.model(timeSeries, modelOrder);
		System.out.println(model.aic()); // Get and display the model AIC
		System.out.println(model.coefficients()); // Get and display the estimated coefficients
		System.out.println(java.util.Arrays.toString(model.stdErrors()));
		plot(model.predictionErrors());
		Forecast forecast = model.forecast(12); // To specify the alpha significance level, add it as a second argument.
		System.out.println(forecast);
	}
}
