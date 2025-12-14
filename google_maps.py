import googlemaps
import os

class GoogleMapsClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("Google Maps API key not found. Set the GOOGLE_MAPS_API_KEY environment variable.")
        self.client = googlemaps.Client(key=self.api_key)

    def get_coordinates(self, address):
        """
        Geocode an address to get its latitude and longitude.
        """
        try:
            geocode_result = self.client.geocode(address)
            if geocode_result:
                lat = geocode_result[0]['geometry']['location']['lat']
                lng = geocode_result[0]['geometry']['location']['lng']
                return lat, lng
        except Exception as e:
            print(f"Error geocoding {address}: {e}")
        return None, None

    def get_distance_matrix(self, origins, destinations):
        """
        Calculate the distance matrix between origins and destinations.
        """
        try:
            distance_matrix = self.client.distance_matrix(origins, destinations, mode="driving")
            return distance_matrix
        except Exception as e:
            print(f"Error getting distance matrix: {e}")
        return None
