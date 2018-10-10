package test;


import org.junit.Test;
import org.numenta.nupic.Parameters;
import org.numenta.nupic.Parameters.KEY;
import org.numenta.nupic.algorithms.Classifier;
import org.numenta.nupic.algorithms.SDRClassifier;
import org.numenta.nupic.encoders.Encoder;
import org.numenta.nupic.util.Tuple;
import org.numenta.nupic.algorithms.Anomaly;
import org.numenta.nupic.algorithms.CLAClassifier;
import org.numenta.nupic.algorithms.SpatialPooler;
import org.numenta.nupic.algorithms.TemporalMemory;
import org.numenta.nupic.datagen.ResourceLocator;
import org.numenta.nupic.encoders.MultiEncoder;
import org.numenta.nupic.network.Inference;
import org.numenta.nupic.network.Layer;
import org.numenta.nupic.network.Network;
import org.numenta.nupic.network.Region;
import org.numenta.nupic.network.sensor.FileSensor;
import org.numenta.nupic.network.sensor.Sensor;
import org.numenta.nupic.network.sensor.SensorParams;
import org.numenta.nupic.network.sensor.SensorParams.Keys;
import org.numenta.nupic.examples.napi.hotgym.NetworkDemoHarness;
import org.numenta.nupic.model.Connections;

class HTMNAPIQuickStart {

	@Test
	public void NAPIQuickStartSingleLayer()
	{
		Parameters p = NetworkDemoHarness.getParameters(); // "Default" test parameters (you will need to tweak)
		p = p.union(NetworkDemoHarness.getNetworkDemoTestEncoderParams()); // Combine "default" encoder parameters.

		Network network = Network.create("Network API Demo", p)         // Name the Network whatever you wish...
		    .add(Network.createRegion("Region 1")                       // Name the Region whatever you wish...
		        .add(Network.createLayer("Layer 2/3", p)                // Name the Layer whatever you wish...
		            .alterParameter(KEY.AUTO_CLASSIFY, Boolean.TRUE)    // (Optional) Add a CLAClassifier
		            .add(Anomaly.create())                              // (Optional) Add an Anomaly detector
		            .add(new TemporalMemory())                          // Core Component but also it's "optional"
		            .add(new SpatialPooler())                           // Core Component, but also "optional"
		            .add(Sensor.create(FileSensor::create, SensorParams.create(
		                Keys::path, "", ResourceLocator.path("rec-center-hourly.csv"))))));  // Sensors automatically connect to your source data, but you may omit this and pump data direction in!

		network.start();
	}
	
	@Test
	public void NAPIQuickStartMultiLayer()
	{
		Parameters p = NetworkDemoHarness.getParameters();
		p = p.union(NetworkDemoHarness.getNetworkDemoTestEncoderParams());
		        
		Network network = Network.create("Network API Demo", p)
		    .add(Network.createRegion("Region 1")
		        .add(Network.createLayer("Layer 2/3", p)
		            .alterParameter(KEY.AUTO_CLASSIFY, Boolean.TRUE)
		            .add(Anomaly.create())
		            .add(new TemporalMemory()))
		        .add(Network.createLayer("Layer 4", p)
		            .add(new SpatialPooler()))
		        .add(Network.createLayer("Layer 5", p)
		            .add(Sensor.create(FileSensor::create, SensorParams.create(
		                Keys::path, "", ResourceLocator.path("rec-center-hourly.csv")))))
		        .connect("Layer 2/3", "Layer 4")
		        .connect("Layer 4", "Layer 5"));
	}
	
	@Test
	public void NAPIQuickStartMultiRegion()
	{
		Parameters p = NetworkDemoHarness.getParameters();
		p = p.union(NetworkDemoHarness.getNetworkDemoTestEncoderParams());

		// Shared connections example
		Connections connections = new Connections();
		        
		Network network = Network.create("Network API Demo", p)
		    .add(Network.createRegion("Region 1")
		        .add(Network.createLayer("Layer 2/3", p)
		            .alterParameter(KEY.AUTO_CLASSIFY, Boolean.TRUE)
		            .using(connections)       // Demonstrates Connections sharing between Layers in same Region
		            .add(Anomaly.create())
		            .add(new TemporalMemory()))
		        .add(Network.createLayer("Layer 4", p)
		            .using(connections)       // Shared with different Layer above
		            .add(new SpatialPooler()))
		        .connect("Layer 2/3", "Layer 4"))
		    .add(Network.createRegion("Region 2")
		        .add(Network.createLayer("Layer 2/3", p)
		            .alterParameter(KEY.AUTO_CLASSIFY, Boolean.TRUE)
		            .add(Anomaly.create())
		            .add(new TemporalMemory())
		            .add(new SpatialPooler()))
		        .add(Network.createLayer("Layer 4", p)
		            .add(Sensor.create(FileSensor::create, SensorParams.create(
		                Keys::path, "", ResourceLocator.path("rec-center-hourly.csv")))))
		        .connect("Layer 2/3", "Layer 4"))
		    .connect("Region 1", "Region 2");
	}

}
