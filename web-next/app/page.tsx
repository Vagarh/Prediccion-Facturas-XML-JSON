import Nav from "@/components/Nav";
import Hero from "@/components/Hero";
import SectionDivider from "@/components/SectionDivider";
import Problema from "@/components/Problema";
import Solucion from "@/components/Solucion";
import Resultados from "@/components/Resultados";
import Drift from "@/components/Drift";
import Explicabilidad from "@/components/Explicabilidad";
import Arquitectura from "@/components/Arquitectura";
import FuturoUI from "@/components/FuturoUI";
import VideoSection from "@/components/VideoSection";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <SectionDivider />
        <Problema />
        <SectionDivider />
        <Solucion />
        <SectionDivider />
        <Resultados />
        <SectionDivider />
        <Drift />
        <SectionDivider />
        <Explicabilidad />
        <SectionDivider />
        <Arquitectura />
        <SectionDivider />
        <FuturoUI />
        <SectionDivider />
        <VideoSection />
      </main>
      <Footer />
    </>
  );
}
