
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import StrategyForm from './StrategyForm';
import MT5ConnectionForm from './MT5ConnectionForm';
import { MT5Credentials, StrategyConfig } from '@/lib/types';
import { useToast } from '@/components/ui/use-toast';

const Dashboard = () => {
  const { toast } = useToast();
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [strategies, setStrategies] = useState<StrategyConfig[]>([]);

  const handleConnect = async (credentials: MT5Credentials) => {
    try {
      setIsConnecting(true);
      // This would make an API call to your backend
      console.log('Connecting to MT5 with credentials:', credentials);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setIsConnected(true);
      toast({
        title: "Connected to MT5",
        description: `Successfully connected to account ${credentials.accountId} on server ${credentials.server}`,
      });
    } catch (error) {
      toast({
        title: "Connection Failed",
        description: "Failed to connect to MT5. Please check your credentials and try again.",
        variant: "destructive",
      });
    } finally {
      setIsConnecting(false);
    }
  };

  const handleCreateStrategy = async (data: StrategyConfig) => {
    try {
      // This would make an API call to your backend
      console.log('Creating strategy with data:', data);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const newStrategy = {
        ...data,
        id: `strategy-${strategies.length + 1}`,
        isActive: false
      };
      
      setStrategies([...strategies, newStrategy]);
      
      toast({
        title: "Strategy Created",
        description: `Successfully created strategy "${data.name}"`,
      });
      
      return true;
    } catch (error) {
      toast({
        title: "Strategy Creation Failed",
        description: "Failed to create strategy. Please try again.",
        variant: "destructive",
      });
      return false;
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Forex Correlation Strategy Dashboard</h1>
      
      <Tabs defaultValue="connection" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="connection">MT5 Connection</TabsTrigger>
          <TabsTrigger value="strategies" disabled={!isConnected}>Strategies</TabsTrigger>
          <TabsTrigger value="monitor" disabled={!isConnected || strategies.length === 0}>Monitor</TabsTrigger>
        </TabsList>
        
        <TabsContent value="connection">
          <Card>
            <CardHeader>
              <CardTitle>MetaTrader 5 Connection</CardTitle>
              <CardDescription>
                Connect to your MT5 account to start trading
              </CardDescription>
            </CardHeader>
            <CardContent>
              <MT5ConnectionForm 
                onSubmit={handleConnect} 
                isSubmitting={isConnecting}
              />
            </CardContent>
            <CardFooter>
              {isConnected && (
                <div className="text-green-500 font-medium">
                  Connected to MT5
                </div>
              )}
            </CardFooter>
          </Card>
        </TabsContent>
        
        <TabsContent value="strategies">
          <Card>
            <CardHeader>
              <CardTitle>Create Trading Strategy</CardTitle>
              <CardDescription>
                Configure your correlation-based trading strategy
              </CardDescription>
            </CardHeader>
            <CardContent>
              <StrategyForm onSubmit={handleCreateStrategy} />
            </CardContent>
          </Card>
          
          {strategies.length > 0 && (
            <div className="mt-6">
              <h2 className="text-xl font-bold mb-4">Your Strategies</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {strategies.map((strategy) => (
                  <Card key={strategy.id}>
                    <CardHeader>
                      <CardTitle>{strategy.name}</CardTitle>
                      <CardDescription>
                        {strategy.pair1} and {strategy.pair2} on {strategy.timeframe} timeframe
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="text-sm">
                        <p>Correlation Window: {strategy.correlation_window}</p>
                        <p>RSI Window: {strategy.rsi_window}</p>
                        <p>Entry Threshold: {strategy.correlation_entry_threshold}</p>
                        <p>Exit Threshold: {strategy.correlation_exit_threshold}</p>
                      </div>
                    </CardContent>
                    <CardFooter className="flex justify-between">
                      <div className={`text-sm font-medium ${strategy.isActive ? 'text-green-500' : 'text-gray-500'}`}>
                        {strategy.isActive ? 'Active' : 'Inactive'}
                      </div>
                      <div className="flex space-x-2">
                        <button className="text-sm text-blue-500 hover:underline">Edit</button>
                        <button className="text-sm text-red-500 hover:underline">Delete</button>
                        <button className={`text-sm ${strategy.isActive ? 'text-amber-500' : 'text-green-500'} hover:underline`}>
                          {strategy.isActive ? 'Stop' : 'Start'}
                        </button>
                      </div>
                    </CardFooter>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="monitor">
          <Card>
            <CardHeader>
              <CardTitle>Strategy Monitor</CardTitle>
              <CardDescription>
                Monitor your active trading strategies
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                This is where you would see live updates of your strategies.
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Dashboard;
