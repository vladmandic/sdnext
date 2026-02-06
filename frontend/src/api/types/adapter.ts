export interface IPAdapterUnit {
  adapter: string;
  scale: number;
  crop: boolean;
  start: number;
  end: number;
  images: File[];
  masks: File[];
}
