import { motion } from 'framer-motion';
import { Clock, TrendingUp, Users, Zap, Leaf } from 'lucide-react';

const vehicleIcons = {
    Standard: '🚗',
    Premium: '💎',
    Eco_Saver: '🌿',
    Eco: '🌿',
};

const vehicleColors = {
    Standard: 'border-blue-200 hover:border-blue-400 bg-blue-50',
    Premium: 'border-purple-200 hover:border-purple-400 bg-purple-50',
    Eco_Saver: 'border-green-200 hover:border-green-400 bg-green-50',
    Eco: 'border-green-200 hover:border-green-400 bg-green-50',
};

const vehicleDetails = {
    Standard: {
        icon: Users,
        capacity: '4 seats',
        description: 'Comfortable & Affordable',
    },
    Premium: {
        icon: Zap,
        capacity: '4 seats',
        description: 'Luxury Ride Experience',
    },
    Eco_Saver: {
        icon: Leaf,
        capacity: '4 seats',
        description: 'Budget Friendly',
    },
    Eco: {
        icon: Leaf,
        capacity: '4 seats',
        description: 'Budget Friendly',
    },
};

export default function VehicleCard({ vehicle, index, onBook, isRecommended }) {
    const Icon = vehicleDetails[vehicle.type]?.icon || Users;
    const colorClass = vehicleColors[vehicle.type] || vehicleColors.Standard;
    const emoji = vehicleIcons[vehicle.type] || '🚗';
    const details = vehicleDetails[vehicle.type] || vehicleDetails.Standard;

    const surgeColor = vehicle.surge > 1.0 ? 'text-red-500' : 'text-ola-primary';

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            whileHover={{ scale: 1.02, y: -4 }}
            whileTap={{ scale: 0.98 }}
            className={`relative border-2 rounded-2xl p-4 cursor-pointer transition-all ${colorClass}`}
            onClick={() => onBook(vehicle)}
        >
            {/* Recommended Badge */}
            {isRecommended && (
                <div className="absolute -top-3 -right-3 bg-ola-primary text-white text-xs font-bold px-3 py-1 rounded-full shadow-lg z-10">
                    ⭐ Best Match
                </div>
            )}

            <div className="flex items-center justify-between">
                {/* Left: Vehicle Info */}
                <div className="flex items-center gap-4 flex-1">
                    {/* Vehicle Icon */}
                    <div className="text-4xl animate-car-bounce">
                        {emoji}
                    </div>

                    {/* Details */}
                    <div className="flex-1">
                        <div className="flex items-center gap-2">
                            <h3 className="font-bold text-lg text-gray-800">
                                {vehicle.type.replace('_', ' ')}
                            </h3>
                            {vehicle.surge > 1.0 && (
                                <span className="text-xs bg-red-100 text-red-600 px-2 py-0.5 rounded-full font-semibold">
                                    {vehicle.surge}x Surge
                                </span>
                            )}
                        </div>

                        <p className="text-xs text-gray-600 mt-0.5">{details.description}</p>

                        <div className="flex items-center gap-3 mt-2 text-xs text-gray-700">
                            <span className="flex items-center gap-1">
                                <Clock size={14} className="text-ola-primary" />
                                {vehicle.eta} mins
                            </span>
                            <span className="flex items-center gap-1">
                                <Icon size={14} className="text-gray-500" />
                                {details.capacity}
                            </span>
                        </div>
                    </div>
                </div>

                {/* Right: Price & Book Button */}
                <div className="flex flex-col items-end gap-2 ml-4">
                    <div className="text-right">
                        <div className="text-2xl font-black text-gray-900">
                            ₹{vehicle.price}
                        </div>
                        {vehicle.surge > 1.0 && (
                            <div className="text-xs text-red-500 font-medium">
                                +₹{((vehicle.price / vehicle.surge) * (vehicle.surge - 1)).toFixed(0)} surge
                            </div>
                        )}
                    </div>

                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="bg-ola-primary hover:bg-ola-dark text-white font-bold px-6 py-2 rounded-full text-sm shadow-lg transition"
                    >
                        Book Now
                    </motion.button>
                </div>
            </div>
        </motion.div>
    );
}
