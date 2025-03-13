
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
import { MT5Credentials } from '@/lib/types';
import { useToast } from '@/components/ui/use-toast';

const formSchema = z.object({
  accountId: z.string().min(1, { message: 'Account ID is required' }),
  password: z.string().min(1, { message: 'Password is required' }),
  server: z.string().min(1, { message: 'Server is required' }),
  terminalId: z.string().optional(),
});

type MT5ConnectionFormProps = {
  defaultValues?: Partial<MT5Credentials>;
  onSubmit: (data: z.infer<typeof formSchema>) => void;
  isSubmitting?: boolean;
};

export function MT5ConnectionForm({ defaultValues, onSubmit, isSubmitting = false }: MT5ConnectionFormProps) {
  const { toast } = useToast();
  
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: defaultValues || {
      accountId: '',
      password: '',
      server: '',
      terminalId: '',
    },
  });

  function handleSubmit(values: z.infer<typeof formSchema>) {
    try {
      onSubmit(values);
      toast({
        title: "Connection initiated",
        description: "Attempting to connect to MT5...",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to connect to MT5",
        variant: "destructive",
      });
    }
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-4">
        <FormField
          control={form.control}
          name="accountId"
          render={({ field }) => (
            <FormItem>
              <FormLabel>MT5 Account ID</FormLabel>
              <FormControl>
                <Input placeholder="12345678" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        
        <FormField
          control={form.control}
          name="password"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Password</FormLabel>
              <FormControl>
                <Input type="password" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        
        <FormField
          control={form.control}
          name="server"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Server</FormLabel>
              <FormControl>
                <Input placeholder="broker-server" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        
        <FormField
          control={form.control}
          name="terminalId"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Terminal ID (optional)</FormLabel>
              <FormControl>
                <Input placeholder="MT5 terminal identifier" {...field} />
              </FormControl>
              <FormDescription>Optional identifier for your MT5 terminal</FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? 'Connecting...' : 'Connect to MT5'}
        </Button>
      </form>
    </Form>
  );
}

export default MT5ConnectionForm;
