
import React from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Button } from '@/components/ui/button';
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { StrategyConfig } from '@/lib/types';
import { useToast } from '@/components/ui/use-toast';

const formSchema = z.object({
  name: z.string().min(2, {
    message: 'Strategy name must be at least 2 characters.',
  }),
  pair1: z.string().min(3, {
    message: 'First pair is required',
  }),
  pair2: z.string().min(3, {
    message: 'Second pair is required',
  }),
  timeframe: z.enum(['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'], {
    required_error: 'Please select a timeframe.',
  }),
  correlation_window: z.coerce.number().int().positive(),
  rsi_window: z.coerce.number().int().positive(),
  rsi_overbought: z.coerce.number().min(50).max(100),
  rsi_oversold: z.coerce.number().min(0).max(50),
  correlation_entry_threshold: z.coerce.number().min(-1).max(1),
  correlation_exit_threshold: z.coerce.number().min(-1).max(1),
  cooldown_period: z.coerce.number().positive(),
  lotsize_pair1: z.coerce.number().positive(),
  lotsize_pair2: z.coerce.number().positive(),
});

type StrategyFormProps = {
  defaultValues?: Partial<StrategyConfig>;
  onSubmit: (data: z.infer<typeof formSchema>) => void;
  isSubmitting?: boolean;
};

export function StrategyForm({ defaultValues, onSubmit, isSubmitting = false }: StrategyFormProps) {
  const { toast } = useToast();
  
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: defaultValues || {
      name: 'New Strategy',
      pair1: 'EURUSD',
      pair2: 'GBPUSD',
      timeframe: '1h',
      correlation_window: 50,
      rsi_window: 14,
      rsi_overbought: 60,
      rsi_oversold: 40,
      correlation_entry_threshold: -0.3,
      correlation_exit_threshold: 0.7,
      cooldown_period: 24,
      lotsize_pair1: 0.1,
      lotsize_pair2: 0.01,
    },
  });

  function handleSubmit(values: z.infer<typeof formSchema>) {
    try {
      onSubmit(values);
      toast({
        title: "Strategy saved",
        description: "Your strategy settings have been saved",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to save strategy",
        variant: "destructive",
      });
    }
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-4">
        <FormField
          control={form.control}
          name="name"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Strategy Name</FormLabel>
              <FormControl>
                <Input placeholder="My Strategy" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        
        <div className="grid grid-cols-2 gap-4">
          <FormField
            control={form.control}
            name="pair1"
            render={({ field }) => (
              <FormItem>
                <FormLabel>First Pair</FormLabel>
                <FormControl>
                  <Input placeholder="EURUSD" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          
          <FormField
            control={form.control}
            name="pair2"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Second Pair</FormLabel>
                <FormControl>
                  <Input placeholder="GBPUSD" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>

        <FormField
          control={form.control}
          name="timeframe"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Timeframe</FormLabel>
              <Select onValueChange={field.onChange} defaultValue={field.value}>
                <FormControl>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a timeframe" />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  <SelectItem value="1m">1 Minute</SelectItem>
                  <SelectItem value="5m">5 Minutes</SelectItem>
                  <SelectItem value="15m">15 Minutes</SelectItem>
                  <SelectItem value="30m">30 Minutes</SelectItem>
                  <SelectItem value="1h">1 Hour</SelectItem>
                  <SelectItem value="4h">4 Hours</SelectItem>
                  <SelectItem value="1d">1 Day</SelectItem>
                  <SelectItem value="1w">1 Week</SelectItem>
                  <SelectItem value="1M">1 Month</SelectItem>
                </SelectContent>
              </Select>
              <FormMessage />
            </FormItem>
          )}
        />

        <div className="grid grid-cols-2 gap-4">
          <FormField
            control={form.control}
            name="correlation_window"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Correlation Window</FormLabel>
                <FormControl>
                  <Input type="number" {...field} />
                </FormControl>
                <FormDescription>Number of periods for correlation calculation</FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />
          
          <FormField
            control={form.control}
            name="rsi_window"
            render={({ field }) => (
              <FormItem>
                <FormLabel>RSI Window</FormLabel>
                <FormControl>
                  <Input type="number" {...field} />
                </FormControl>
                <FormDescription>Number of periods for RSI calculation</FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <FormField
            control={form.control}
            name="rsi_overbought"
            render={({ field }) => (
              <FormItem>
                <FormLabel>RSI Overbought</FormLabel>
                <FormControl>
                  <Input type="number" {...field} />
                </FormControl>
                <FormDescription>RSI threshold for overbought condition</FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />
          
          <FormField
            control={form.control}
            name="rsi_oversold"
            render={({ field }) => (
              <FormItem>
                <FormLabel>RSI Oversold</FormLabel>
                <FormControl>
                  <Input type="number" {...field} />
                </FormControl>
                <FormDescription>RSI threshold for oversold condition</FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <FormField
            control={form.control}
            name="correlation_entry_threshold"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Correlation Entry Threshold</FormLabel>
                <FormControl>
                  <Input type="number" step="0.1" {...field} />
                </FormControl>
                <FormDescription>Threshold for trade entry (-1.0 to 1.0)</FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />
          
          <FormField
            control={form.control}
            name="correlation_exit_threshold"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Correlation Exit Threshold</FormLabel>
                <FormControl>
                  <Input type="number" step="0.1" {...field} />
                </FormControl>
                <FormDescription>Threshold for trade exit (-1.0 to 1.0)</FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>

        <FormField
          control={form.control}
          name="cooldown_period"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Cooldown Period (hours)</FormLabel>
              <FormControl>
                <Input type="number" {...field} />
              </FormControl>
              <FormDescription>Minimum time between trades (hours)</FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        <div className="grid grid-cols-2 gap-4">
          <FormField
            control={form.control}
            name="lotsize_pair1"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Lot Size Pair 1</FormLabel>
                <FormControl>
                  <Input type="number" step="0.01" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          
          <FormField
            control={form.control}
            name="lotsize_pair2"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Lot Size Pair 2</FormLabel>
                <FormControl>
                  <Input type="number" step="0.01" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>

        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? 'Saving...' : 'Save Strategy'}
        </Button>
      </form>
    </Form>
  );
}

export default StrategyForm;
