
public class Kinematics {
	
	public double initial_height; //how high above the ground the ball starts

	public double initial_velocity; //how fast the ball is moving at the beginning of its motion
	
	public double angle_in_degrees; //the angle the ball makes with the horizontal at the beginning of its motion (deg)
	
	public double angle_in_radians; //the angle the ball makes with the horizontal at the beginning of its motion (rad)
	
	public double predicted_falling_distance; //how fall the ball is expected to fall before it is in the same plane as the net
	
	public double probability_of_success; //work in progress
	
	public double x_distance_change; //how much the ball moves horizontally in a short interval near the beginning of its motion
	
	public double y_distance_change; //how much the ball moves vertically in a short interval near the beginning of its motion
	
	public static double distance_from_net_horizontal; //how far away from the net the ball is hit
	
	public static double vertical_acceleration = -9.807; //acceleration due to gravity on Earth
	
	public static double net_height = 1; //how tall the net is (presumed to be one meter)
	
	public double x_velocity; //how fast the ball is moving horizontally (should remain nearly constant because projectile motion)
	
	public double y_velocity_initial; //how fast the ball is moving vertically immediately after it is hit
	
	public double predicted_time_to_reach_net; //the predicted time it will take for the ball to be in the same plane as the net
	
	public double predicted_clearing_distance = -1; //how high above the net the ball is expected to travel. Initially set to -1, meaning the ball could not pass over, and changed later if it can
	
	public static void main(String[] args) { //Creates an example, instantiating the needed variables and finding how much it will clear the net by. In the given example, it should clear the net by 0.575 meters.
		Kinematics k = new Kinematics();
		k.x_distance_change = Math.sqrt(3);
		k.y_distance_change = -1;
		k.initial_velocity = 5;
		k.initial_height = 11;
		distance_from_net_horizontal = 5;
		k.fixVariables();
		k.willTheBallClear();
	}
	
	/* This method uses the known variables to calculate unknown ones.
	 * The angle is found with the arctangent of a small y change over a small x change.
	 * The x velocity is found by multiplying the initial velocity by the cosine of the angle the ball is hit at.
	 * The initial y velocity is found by multiplying the initial velocity by the sine of the angle the ball is hit at.
	 * The predicted time to reach the net is found by dividing how far the ball is initially away from the net by its x-velocity.
	 * The predicted distance the ball falls if found with the Uniformly Accelerated Motion equation: x=v_i*t+1/2*a*t^2, or in other words: position final=initial velocity multiplied by time plus half of acceleration times time squared.
	 * The predicted clearing distance is calculated, and is -1 if it is predicted to not pass the net */
	public void fixVariables () {
		angle_in_degrees = Math.atan(this.y_distance_change/this.x_distance_change) * 180 / Math.PI;
		angle_in_radians = this.angle_in_degrees * Math.PI / 180;
		x_velocity = this.initial_velocity * Math.cos(this.angle_in_radians);
		y_velocity_initial = this.initial_velocity * Math.sin(this.angle_in_radians);
		predicted_time_to_reach_net = distance_from_net_horizontal / this.x_velocity;
		predicted_falling_distance = Math.abs(this.y_velocity_initial * this.predicted_time_to_reach_net + vertical_acceleration * Math.pow(this.predicted_time_to_reach_net, 2) * 1/2);
		predicted_clearing_distance = (this.initial_height - this.predicted_falling_distance >= net_height?  this.initial_height - this.predicted_falling_distance - net_height : -1);
	}
	
	/* This method prints to the console whether or not the ball is expected to clear the net.
	 * In order to use this function, it is necessary to instantiate the needed variables and run the fixVariables method.
	 * If the ball is expected to clear the net, it will print to the console the expected clearing distance, rounded to three digits after the decimal. */
	public void willTheBallClear () {
		System.out.println("The ball " + ((this.predicted_clearing_distance < 0)? "is predicted to not clear the net":"is predicted to clear the net by " + ((double) ((int) (1000 * this.predicted_clearing_distance))) / 1000 + " meters."));
	}

}
